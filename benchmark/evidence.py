"""Strictly audit the archived benchmark and generate its evidence report.

This checks internal consistency, not the historical authenticity of a run.
Unlike Python assertions, these checks also run under ``python -O``.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
MODES = ("raw", "smooth", "project", "feedback")
SWITCHES = {
    "raw": (False, False, False, False, False),
    "smooth": (True, False, False, False, False),
    "project": (False, True, False, True, False),
    "feedback": (False, True, True, True, True),
}


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def number(value):
    require(not isinstance(value, bool), "Boolean used as a number")
    result = float(value)
    require(math.isfinite(result), "Nonfinite number")
    return result


def integer(value):
    result = number(value)
    require(result.is_integer(), f"Expected integer: {value}")
    return int(result)


def strict_json(text):
    def reject(value):
        raise EvidenceError(f"Invalid JSON number: {value}")
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(text, parse_constant=reject, object_pairs_hook=pairs)


def frozen_scenes():
    scenes = []
    for yaw in (-5., 0., 5.):
        for tx, ty in ((.265, .105), (.285, .105), (.265, .145), (.285, .145)):
            for size in (.97, 1.03):
                sid = len(scenes)
                scenes.append(dict(scene_id=sid, start_x=.28 + (.004 if sid % 2 else -.004),
                    start_y=-.035, target_x=tx, target_y=ty, yaw_deg=yaw, size_scale=size,
                    mass_kg=.35 + .03 * (sid % 5), pose_delay_s=.06, action_delay_s=.04,
                    differential_noise_m=.009, common_bias_m=.004,
                    dropout_start_s=3., dropout_duration_s=.16))
    return scenes


def seed_for(scene_id, repeat):
    return 170000 + (scene_id + 10) * 10 + repeat


def load_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
                "Duplicate or missing CSV header")
        rows = []
        for raw in reader:
            require(None not in raw and all(v is not None for v in raw.values()), "Malformed CSV row")
            row = dict(raw)
            require(row.get("success") in ("True", "False"), "success must be True or False")
            row["success"] = row["success"] == "True"
            for key in ("scene_id", "repeat", "seed", "steps", "drift_sample_count"):
                row[key] = integer(row[key])
            for key in ("target_error_m", "max_lift_m", "placed_hold_s", "dual_contact_s",
                        "mass_kg", "mean_tracking_m", "upright", "speed_mps"):
                row[key] = number(row[key])
            row["grasp_drift_m"] = number(row["grasp_drift_m"]) if row["grasp_drift_m"] else None
            rows.append(row)
    return rows


def validate_design(manifest, rows):
    require(manifest["suite"] == "frozen_evaluation", "Not the frozen evaluation suite")
    require(manifest["modes"] == list(MODES) and manifest["repetitions"] == 2,
            "Frozen benchmark needs exactly four modes and two repeats")
    scenes = manifest["scenarios"]
    require(len(scenes) == 24, "Expected 24 scenes")
    for actual, expected in zip(scenes, frozen_scenes()):
        require(set(actual) == set(expected), "Unexpected scene fields")
        for key, value in expected.items():
            require(math.isclose(number(actual[key]), value, rel_tol=0, abs_tol=1e-12),
                    f"Scene {expected['scene_id']} changed: {key}")
    expected_keys = {(s, r, m) for s in range(24) for r in range(2) for m in MODES}
    keys = [(r["scene_id"], r["repeat"], r["mode"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate episode identity")
    require(set(keys) == expected_keys and len(rows) == 192, "Missing or unexpected episode")
    for row in rows:
        sid = row["scene_id"]
        require(row["seed"] == seed_for(sid, row["repeat"]), f"Incorrect paired seed: {sid}")
        require(math.isclose(row["mass_kg"], scenes[sid]["mass_kg"], abs_tol=1e-12), "Mass mismatch")
        require(row["steps"] > 0 and row["drift_sample_count"] >= 0, "Invalid sample counts")
        require(row["target_error_m"] >= 0 and row["speed_mps"] >= 0, "Negative error/speed")
        require((row["grasp_drift_m"] is None) == (row["drift_sample_count"] == 0), "Missing drift samples")
        if row["success"]:
            require(row["max_lift_m"] >= .1 and row["dual_contact_s"] >= .5
                    and row["placed_hold_s"] >= 1. and row["failure"] == "none", "Inconsistent success")
        else:
            require(row["failure"] != "none", "Failed episode missing failure reason")


def wilson(k, n):
    z = 1.96
    p, denominator = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [100 * (center - radius), 100 * (center + radius)]


def mode_statistics(rows):
    result = {}
    for mode in MODES:
        group = [r for r in rows if r["mode"] == mode]
        n, k = len(group), sum(r["success"] for r in group)
        require(n > 0, f"Empty mode: {mode}")
        drift = [r["grasp_drift_m"] for r in group if r["grasp_drift_m"] is not None]
        result[mode] = dict(episodes=n, successes=k, success_pct=100 * k / n,
            wilson95_pct=wilson(k, n), mean_target_error_cm=100 * sum(r["target_error_m"] for r in group) / n,
            mean_grasp_drift_cm=100 * sum(drift) / len(drift) if drift else None,
            drift_eligible_episodes=len(drift), failures=dict(Counter(r["failure"] for r in group)))
    return result


def paired_statistics(rows, baseline, treatment, resamples=10000):
    indexed = {(r["scene_id"], r["repeat"], r["mode"]): r["success"] for r in rows}
    counts = dict(both_success=0, baseline_only=0, treatment_only=0, both_failure=0)
    contrasts = []
    for sid in range(24):
        change = 0
        for repeat in range(2):
            a, b = indexed[sid, repeat, baseline], indexed[sid, repeat, treatment]
            counts["both_success" if a and b else "baseline_only" if a else "treatment_only" if b else "both_failure"] += 1
            change += int(b) - int(a)
        contrasts.append(change)
    # Exact paired scene sign-flip distribution, with the two repeats kept together.
    distribution = Counter({0: 1})
    for delta in contrasts:
        next_distribution = Counter()
        for total, ways in distribution.items():
            next_distribution[total + delta] += ways
            next_distribution[total - delta] += ways
        distribution = next_distribution
    observed = abs(sum(contrasts))
    p = sum(ways for total, ways in distribution.items() if abs(total) >= observed) / 2 ** len(contrasts)
    rng = random.Random(817)
    boot = sorted(50 * sum(rng.choices(contrasts, k=24)) / 24 for _ in range(resamples))
    def quantile(fraction):
        index = (len(boot) - 1) * fraction
        lo = int(index)
        return boot[lo] + (boot[min(lo + 1, len(boot) - 1)] - boot[lo]) * (index - lo)
    return dict(**counts, pairs=48, clusters=24, success_difference_pp=50 * sum(contrasts) / 24,
                scene_bootstrap95_pp=[quantile(.025), quantile(.975)], bootstrap_seed=817,
                bootstrap_resamples=resamples, scene_sign_flip_p=p,
                sign_flip_assumption="Scene contrasts are exchangeable in sign under the null; not randomized treatment assignment.")


def tensor(value, shape):
    if not shape:
        number(value)
    else:
        require(isinstance(value, list) and len(value) == shape[0], f"Invalid trace shape: {shape}")
        for item in value:
            tensor(item, shape[1:])


def validate_trace(path):
    last, count = -math.inf, 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = strict_json(line)
            stamp = number(row["t"])
            require(stamp >= 0 and stamp > last, f"Unordered trace: {path.name}")
            last, count = stamp, count + 1
            for key, shape in (("q", (14,)), ("command", (14,)), ("object", (4, 4)), ("candidates", (2, 4, 4))):
                tensor(row[key], shape)
            require(len(row["contacts"]) == 2 and all(type(v) is bool for v in row["contacts"]), "Invalid contacts")
            if row["obs_timestamp"] is not None:
                require(0 <= number(row["obs_timestamp"]) <= stamp + 1e-6, "Future pose observation")
    require(count > 0, f"Empty trace: {path.name}")
    return count


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root=ROOT):
    root = Path(root).resolve()
    task, paths = root / "work/g1_objectcentric", {}
    results = task / "results"
    manifest = strict_json((results / "manifest.json").read_text(encoding="utf-8"))
    allowed = {"task.py", "geometry.py", "scenarios.py", "actions.py", "evaluate.py", "reference-parameters.json",
               "scene.mjb", "scene.xml", "policy.json"}
    require(set(manifest["files_sha256"]) == allowed, "Source hash inventory changed")
    for name, expected in manifest["files_sha256"].items():
        path = task / name if name not in {"scene.mjb", "scene.xml", "policy.json"} else root / "work/box_transfer" / name
        require(sha256(path) == expected, f"Frozen source/model changed: {name}")
        paths[path.relative_to(root).as_posix()] = expected
    recorded = strict_json((results / "audit.json").read_text(encoding="utf-8"))
    dependencies = {k.replace("\\", "/"): v for k, v in recorded["shared_dependency_sha256"].items()}
    require(set(dependencies) == {"work/box_lift/task.py", "work/box_transfer/task.py"}, "Dependency inventory changed")
    for name, expected in dependencies.items():
        require(sha256(root / name) == expected, f"Changed dependency: {name}")
        paths[name] = expected
    rows = load_rows(results / "episodes.csv")
    validate_design(manifest, rows)
    stats = mode_statistics(rows)
    saved = strict_json((results / "summary.json").read_text(encoding="utf-8"))
    require(saved["episodes"] == 192 and set(saved["modes"]) == set(MODES), "Summary coverage mismatch")
    for mode, values in stats.items():
        for key, expected in values.items():
            actual = saved["modes"][mode][key]
            if isinstance(expected, float):
                require(math.isclose(number(actual), expected, rel_tol=0, abs_tol=1e-9), f"Summary mismatch: {mode}/{key}")
            elif isinstance(expected, list):
                require(len(actual) == len(expected) and all(math.isclose(number(a), b, abs_tol=1e-9) for a, b in zip(actual, expected)), "Wilson interval mismatch")
            else:
                require(actual == expected, f"Summary mismatch: {mode}/{key}")
    expected_names = {f"scene-{r['scene_id']:02d}_repeat-{r['repeat']}_{r['mode']}.jsonl.gz" for r in rows}
    require({p.name for p in (results / "traces").glob("*.jsonl.gz")} == expected_names, "Trace coverage mismatch")
    trace_samples = 0
    for name in sorted(expected_names):
        path = results / "traces" / name
        trace_samples += validate_trace(path)
        paths[path.relative_to(root).as_posix()] = sha256(path)
    for name in ("manifest.json", "episodes.csv", "summary.json", "audit.json"):
        path = results / name
        paths[path.relative_to(root).as_posix()] = sha256(path)
    comparisons = {f"{a}_vs_{b}": paired_statistics(rows, a, b) for a, b in
                   (("raw", "smooth"), ("raw", "project"), ("raw", "feedback"), ("project", "feedback"))}
    old_pair = saved["paired_raw_feedback"]
    new_pair = comparisons["raw_vs_feedback"]
    for old, new in (("both_success", "both_success"), ("raw_only", "baseline_only"),
                     ("feedback_only", "treatment_only"), ("both_failure", "both_failure"), ("success_difference_pp", "success_difference_pp")):
        require(old_pair[old] == new_pair[new], f"Saved paired table mismatch: {old}")
    return dict(schema_version=1, passed=True, episodes=len(rows), trace_files=192, trace_samples=trace_samples,
        modes=stats, comparisons=comparisons, scenarios=manifest["scenarios"], switches=SWITCHES,
        success_recomputed_from_trace=False, historical_authenticity_proven=False,
        limitations=["Archived traces omit fallen/table contact/Z error/speed/settle state; success cannot be independently reconstructed.",
            "Success uses maximum released stable interval, not necessarily terminal stability.",
            "Size and initial X are confounded; mass also changes with scene ID.",
            "Delay/noise/dropout are fixed here, not independent robustness sweeps.",
            "Synthetic CEM action source and simulator pose feedback; no GR00T inference or RGB-D perception.",
            "Hashes establish present consistency, not the date or authorship of historical experiments."],
        resume_claim_check=dict(claimed_raw_pct=64.6, claimed_feedback_pct=75.,
            claimed_difference_pp=10.4, matches_archived_results=False,
            implied_counts_if_denominator_48=[31, 36], note="No archived evidence for these resume values; do not substitute them for saved measurements."),
        files_sha256=paths)


def markdown(report):
    lines = ["# G1 配对实验核查与面试复盘", "", "由 `python -m benchmark.evidence` 从冻结代码、CSV 和 192 条压缩轨迹生成。", "",
        "## 先回答数据口径", "", "当前可核查记录来自 CEM 轨迹代理和 MuJoCo 位姿传感器模拟。不能称为 GR00T 策略评测或 RGB-D 闭环结果。",
        "简历中的 64.6% → 75.0% 在分母为 48 时对应 31/48 → 36/48；仓库当前记录不支持这组数字。", "",
        "| 配置 | 成功次数 / 总次数 | 成功率 | 最终 XY 误差，含失败 | 抓持漂移 / 有效轨迹数 |",
        "|---|---:|---:|---:|---:|"]
    for mode, s in report["modes"].items():
        lines.append(f"| {mode} | {s['successes']}/{s['episodes']} | {s['success_pct']:.1f}% | {s['mean_target_error_cm']:.3f} cm | {s['mean_grasp_drift_cm']:.3f} cm / {s['drift_eligible_episodes']} |")
    lines += ["", "每组分母都是 24 场景 × 2 次重复 = 48。192 是四组总运行数。漂移只统计 CARRY 双掌接触有效轨迹，不能和最终放置误差混用。", "",
        "## 四组开关的真实含义", "", "| 配置 | 平滑 | SE(3) 投影 | 位置反馈 | 释放原点连续 | 动作时效检查 |", "|---|---|---|---|---|---|"]
    for mode, switches in SWITCHES.items():
        lines.append("| " + mode + " | " + " | ".join("开" if v else "关" for v in switches) + " |")
    lines += ["", "这不是 raw → smooth → project → feedback 的逐层叠加：project 和 feedback 没有开启 smooth 分支。四组共用 CEM 参数、IK、关节限位、速度限制、PD、场景和种子。", "",
        "## 场景和种子", "", "主网格：初始 yaw ∈ {-5°, 0°, 5°}；目标 XY ∈ {(0.265,0.105),(0.285,0.105),(0.265,0.145),(0.285,0.145)} m；尺寸比例 ∈ {0.97,1.03}。",
        "还改变了初始 X 和质量，下面完整列出。质量不是独立全因子变量，初始 X 与尺寸绑定；不能据此分别归因。",
        "种子公式：`170000 + (scene_id + 10) * 10 + repeat`。同一场景同一次重复的四组使用完全相同的 seed；两次重复不是新的场景。", "",
        "| 场景 | yaw° | 目标 X/Y m | 尺寸 | 起始 X m | 质量 kg | repeat 0 / 1 seed |", "|---:|---:|---|---:|---:|---:|---|"]
    for s in report["scenarios"]:
        i = s["scene_id"]
        lines.append(f"| {i} | {s['yaw_deg']:g} | {s['target_x']:.3f}/{s['target_y']:.3f} | {s['size_scale']:.2f} | {s['start_x']:.3f} | {s['mass_kg']:.2f} | {seed_for(i,0)} / {seed_for(i,1)} |")
    lines += ["", "所有正式场景固定：位姿延迟 60ms、动作观测延迟 40ms、差异扰动标称 9mm、共同偏置标准差 4mm、3s 起遮挡 160ms。这里没有独立扫描这些变量。", "",
        "## 成功判定及其局限", "", "冻结代码要求最大抬升 ≥10cm、双掌累计接触 ≥0.5s、全过程未触发物体 Z<0.70m 的跌落标志，以及释放后连续稳定区间的最大值 ≥1s。",
        "稳定区间在 100Hz 检查：箱子接触桌面、XY 误差<3.5cm、Z 误差<2cm、Rzz>0.96、速度<3.5cm/s、双掌不接触且阶段为 RETREAT/SETTLE/DONE。",
        "`placed_hold_s` 保存历史最长稳定区间，冻结判定没有要求最后一刻仍满足。因此应准确表述为“曾连续稳定至少1s”。旧轨迹缺少完整判定字段，审计不能独立重算 success；新 runner 会记录完整遥测并同时报告末端稳定判定。", "",
        "## 配对统计与因果归因", "", "| 比较 | 两组都成功 | 仅基线成功 | 仅处理成功 | 两组都失败 | 差值 pp | 场景聚类 bootstrap 95% CI pp |", "|---|---:|---:|---:|---:|---:|---|"]
    for key, p in report["comparisons"].items():
        lo, hi = p["scene_bootstrap95_pp"]
        lines.append(f"| {key} | {p['both_success']} | {p['baseline_only']} | {p['treatment_only']} | {p['both_failure']} | {p['success_difference_pp']:.2f} | [{lo:.2f}, {hi:.2f}] |")
    lines += ["", "bootstrap 以24个场景为单位重采样，保留两次重复和组间配对；固定 seed=817，10,000 次。原始 summary 的4,000次区间也保留，重采样次数不同不表示新增实验。",
        "raw 对 feedback 的区间跨零，不能声称已证明成功率提升。raw 对完整组比较的是一组模块；project 对 feedback 更接近增量反馈效应，但还需检查时效分支等实现差异。",
        "几何投影降低了命令空间残差，但本批成功数从37降到35，不能声称投影本身提高任务成功率。未随机取样的固定场景也不能代表真实任务分布。",
        "严格的后续实验需预先冻结新协议、独立改变质量/初始位姿/延迟/遮挡、留出未见场景、增加独立种子，并报告全部失败和配对区间。不能反复调参直到这24个场景显著。", "",
        "## 核查范围", "", f"已检查 {report['episodes']} 条 CSV、{report['trace_files']} 个轨迹文件、{report['trace_samples']} 个控制记录；检查场景全集、唯一键、seed、源代码/模型散列、统计、配对列联表、时间顺序与有限值。",
        "SHA256 清单写入 JSON。它证明目前文件一致，不能证明历史日期、作者、真机运行或 GR00T 权重来源。", "",
        "复现：`python -m benchmark.evidence --json artifacts/evidence.json --markdown artifacts/evidence.md`。输出必须使用新路径；原始结果不会被覆盖。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    try:
        for path in (args.json, args.markdown):
            if path is not None:
                require(not path.exists(), f"Refusing to overwrite: {path}")
        report = audit(args.root)
        for path, content in ((args.json, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                              (args.markdown, markdown(report))):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(content)
        print(json.dumps({k: report[k] for k in ("passed", "episodes", "trace_files", "trace_samples", "resume_claim_check")}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Audit failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
