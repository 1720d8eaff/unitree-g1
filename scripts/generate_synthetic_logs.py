"""Generate explicitly synthetic resume-aligned log fixtures, never robot evidence.

Outcomes are selected to match prescribed rounded example targets. This is not
a physics simulation, model inference, training run or historical log export.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

SOURCE = "synthetic"
DISCLAIMER = "模拟数据：按简历目标构造的日志示例；未执行真机、GR00T/SmolVLA推理或对应训练，不能作为实验成果。"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def rng_for(seed, key):
    return random.Random(int(sha([seed, key])[:16], 16))


def tagged(**fields):
    return dict(source=SOURCE, synthetic=True, clock="virtual_monotonic", **fields)


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def paired_trials(seed, condition, policies, counts, scenes=18, training_seeds=(0, 1, 2)):
    randomizer = rng_for(seed, [condition, "initial_conditions"])
    cases = []
    for training_seed in training_seeds:
        for scene in range(scenes):
            for repeat in range(2):
                cases.append(dict(seed=training_seed, scene_id=f"syn-scene-{scene:02d}", repeat=repeat,
                    environment_seed=seed + training_seed * 10000 + scene * 10 + repeat,
                    task_variant=("pick_place", "container_drop", "stack")[scene % 3],
                    difficulty=randomizer.random(), initial_condition=dict(x_m=round(randomizer.uniform(.22, .34), 5),
                        y_m=round(randomizer.uniform(-.07, .07), 5), yaw_rad=round(randomizer.uniform(-.2, .2), 5))))
    protocol = sha(dict(kind="synthetic_fixture_protocol", condition=condition, cases=len(cases), seed=seed))
    outcomes = {}
    for policy, successes in zip(policies, counts):
        if not 0 <= successes <= len(cases):
            raise ValueError("Invalid prescribed synthetic success count")
        noise = rng_for(seed, [condition, policy, "scores"])
        ranking = sorted(range(len(cases)), key=lambda i: cases[i]["difficulty"] + noise.gauss(0., .14))
        outcomes[policy] = set(ranking[:successes])
    rows = []
    for i, case in enumerate(cases):
        for policy in policies:
            randomizer = rng_for(seed, [condition, policy, i])
            success = i in outcomes[policy]
            failure = None if success else randomizer.choices(
                ["grasp", "localization", "execution_deviation", "target_change", "timeout"], [30, 15, 27, 16, 12])[0]
            spec = dict(kind="synthetic_policy_spec_not_weights", policy=policy, seed=case["seed"])
            rows.append(tagged(experiment=f"synthetic-{condition}", policy=policy,
                checkpoint=f"synthetic-spec/{policy}/seed-{case['seed']}", checkpoint_sha256=sha(spec),
                checkpoint_hash_kind="hash_of_fictional_policy_spec_not_model_weights",
                task="tabletop_multistage_fixture", condition=condition, seed=case["seed"],
                trial_id=f"syn-{condition}-{case['seed']}-{i:03d}", scene_id=case["scene_id"], repeat=case["repeat"],
                environment_seed=case["environment_seed"], task_variant=case["task_variant"],
                protocol_sha256=protocol, initial_condition_sha256=sha(case["initial_condition"]),
                initial_condition=case["initial_condition"], success=success, failure_type=failure,
                duration_s=round(max(8., randomizer.gauss(27.5 if success else 31., 3.8)), 4)))
    return rows


def group_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"], row["policy"]].append(row)
    result = []
    for (condition, policy), group in sorted(groups.items()):
        successes = sum(r["success"] for r in group)
        result.append(tagged(condition=condition, policy=policy, trials=len(group), successes=successes,
            success_pct=100 * successes / len(group), mean_duration_s=statistics.fmean(r["duration_s"] for r in group),
            failures=dict(Counter(r["failure_type"] for r in group if not r["success"])),
            interpretation="Constructed fixture counts, not an estimate of policy capability."))
    return result


def rescale_mean(values, target):
    multiplier = target / statistics.fmean(values)
    return [value * multiplier for value in values]


def so101_logs(seed):
    policies = ["original", "history", "failure_aware"]
    fixed = paired_trials(seed, "fixed", policies, [69, 78, 84])
    disturbed = paired_trials(seed, "perturbed", policies, [40, 51, 62])
    rows = fixed + disturbed
    # This denominator is prescribed explicitly; recovery is not inferred from final success.
    for policy in policies:
        group = [r for r in disturbed if r["policy"] == policy]
        successes, failures = [r for r in group if r["success"]], [r for r in group if not r["success"]]
        randomizer = rng_for(seed, ["recovery", policy])
        randomizer.shuffle(successes)
        randomizer.shuffle(failures)
        count = {"original": 14, "history": 24, "failure_aware": 33}[policy]
        identified = successes[:count] + failures[:54 - count]
        selected = {id(r) for r in identified}
        recovered = {id(r) for r in successes[:count]}
        for row in group:
            row["recognized_failure"] = id(row) in selected
            row["autonomous_recovery"] = id(row) in recovered
            row["detected_failure_type"] = "grasp" if row["recognized_failure"] else None
    # Fixed trials use explicit negative annotations, avoiding mixed unknown labels.
    for row in fixed:
        row.update(recognized_failure=False, autonomous_recovery=False, detected_failure_type=None)
    dynamic = paired_trials(seed, "dynamic_target", ["sync", "naive_async", "aligned_async"], [52, 61, 69])
    duration_targets = {"sync": 30.6847, "naive_async": 28.5263, "aligned_async": 26.2371}
    for policy, target in duration_targets.items():
        selected = [r for r in dynamic if r["policy"] == policy]
        values = rescale_mean([r["duration_s"] for r in selected], target)
        for row, value in zip(selected, values):
            row["duration_s"] = round(value, 6)
    replans = []
    for policy, target in (("sync", 175.2386), ("naive_async", 96.8421), ("aligned_async", 58.1634)):
        randomizer = rng_for(seed, [policy, "timing"])
        # Warm-up, correlated load variation and occasional scheduling stalls.
        values = [max(1., randomizer.gauss(target, target * .13) + (target * .7 if i < 2 else 0.)
                      + (target * .45 if i in (31, 77, 113) else 0.)) for i in range(144)]
        blocked = rescale_mean(values, target)
        t_ns = 0
        for i, wait in enumerate(blocked):
            capture = t_ns
            preprocess = randomizer.uniform(3.2, 8.9)
            inference = max(15., randomizer.gauss(158., 21.))
            communication = randomizer.uniform(.65, 2.4)
            # Separate planner waits from executor ticks; these are different metrics.
            elapsed = preprocess + inference + communication
            replans.append(tagged(mode=policy, request_id=f"syn-{policy}-{i:04d}",
                measurement_scope="planner_replan_wait_event_not_all_control_ticks", virtual_capture_ns=capture,
                preprocess_ms=round(preprocess, 6), inference_ms=round(inference, 6),
                communication_ms=round(communication, 6), blocked_ms=round(wait, 6),
                action_age_ms=round(max(elapsed + randomizer.uniform(8., 28.),
                                        wait + preprocess + communication + randomizer.uniform(2., 7.)), 6),
                disposition="discarded_stale" if i in (31, 77) and policy != "sync" else "accepted",
                warning="overlap_state_drift" if i == 113 and policy == "aligned_async" else None))
            t_ns += int(randomizer.uniform(.18, .25) * 1e9)
    training = []
    for policy in ("history", "failure_aware"):
        for training_seed in (0, 1, 2):
            randomizer = rng_for(seed, [policy, training_seed, "fictional_training"])
            elapsed = 0.
            for step in range(400, 30001, 400):
                elapsed += randomizer.uniform(49., 61.)
                floor = .022 if policy == "history" else .018
                loss = floor + .19 * math.exp(-step / 6200) + randomizer.gauss(0., .0018)
                validation = floor + .008 + .11 * math.exp(-step / 7600) + randomizer.gauss(0., .0028)
                spike = step in (8400, 20400)
                training.append(tagged(policy=policy, seed=training_seed, step=step, virtual_elapsed_s=round(elapsed, 3),
                    action_loss=round(max(.001, loss + (.009 if spike else 0.)), 7),
                    validation_action_loss=round(max(.001, validation), 7),
                    learning_rate=round(1e-4 * min(1., step / 2400) * (.15 + .85 * (1 + math.cos(math.pi * step / 30000)) / 2), 10),
                    gradient_norm=round(abs(randomizer.gauss(.78 if not spike else 2.4, .19)), 5),
                    train_episode_count=360, corrective_episode_count=140,
                    count_scope="prescribed fictional recipe, no dataset files or training run exist"))
    curves = []
    # An illustrative threshold crossing, explicitly prescribed rather than learned.
    curve_counts = {"original": [18, 21, 23, 24, 26], "failure_aware": [21, 24, 26, 27, 28]}
    for policy, successes in curve_counts.items():
        for examples, k in zip((80, 120, 165, 200, 260), successes):
            for training_seed, offset in ((0, -1), (1, 0), (2, 1)):
                curves.append(tagged(policy=policy, seed=training_seed, demonstration_count=examples,
                    trials=36, successes=k + offset, success_pct=100 * (k + offset) / 36,
                    interpretation="Prescribed synthetic learning-curve fixture, not measured sample efficiency."))
    recovery_group = [r for r in disturbed if r["policy"] == "failure_aware" and r["recognized_failure"]]
    summary = tagged(disclaimer=DISCLAIMER, policy_groups=group_summary(rows), async_groups=group_summary(dynamic),
        recovery=dict(recognized_failures=len(recovery_group), autonomous_recoveries=sum(r["autonomous_recovery"] for r in recovery_group),
                      recovery_pct=100 * sum(r["autonomous_recovery"] for r in recovery_group) / len(recovery_group)),
        replan_groups=[dict(mode=mode, samples=len(group), mean_blocked_ms=statistics.fmean(r["blocked_ms"] for r in group),
                           p95_blocked_ms=sorted(r["blocked_ms"] for r in group)[int(.95 * (len(group) - 1))],
                           discarded=sum(r["disposition"] != "accepted" for r in group))
                      for mode in ("sync", "naive_async", "aligned_async")
                      for group in [[r for r in replans if r["mode"] == mode]]],
        sample_efficiency=dict(threshold_pct=72., original_examples=260, failure_aware_examples=165,
                               reduction_pct=100 * (260 - 165) / 260, interpretation="Crossing prescribed in the fixture."))
    return {"policy_trials.jsonl": rows, "async_trials.jsonl": dynamic, "replan_events.jsonl": replans,
            "training.jsonl": training, "learning_curve.jsonl": curves}, summary


def g1_logs(seed):
    policies = ["raw", "smooth", "project", "feedback"]
    rows = paired_trials(seed, "g1_control_recipe", policies, [31, 32, 34, 36], scenes=24, training_seeds=(0,))
    events = []
    for policy, drift_target in (("raw", 2.4037), ("smooth", 2.3541), ("project", 2.1836), ("feedback", 2.0982)):
        selected = [r for r in rows if r["policy"] == policy]
        randomizer = rng_for(seed, [policy, "g1_drift"])
        eligible = [r for r in selected if r["failure_type"] not in ("grasp", "localization")]
        values = rescale_mean([randomizer.uniform(.7, 4.6) for _ in eligible], drift_target)
        for row in selected:
            row["grasp_drift_cm"] = None
        for row, value in zip(eligible, values):
            row["grasp_drift_cm"] = round(value, 6)
        for row in selected:
            row["action_source"] = "synthetic_log_recipe_no_CEM_or_GR00T_execution"
            row["duration_s"] = round(randomizer.uniform(8.4, 11.2), 5)
            phases = ["REACH", "GRIP", "LIFT", "CARRY", "LOWER", "RELEASE", "SETTLE"]
            if row["failure_type"] in ("grasp", "localization"):
                phases = phases[:2]
            weights = [randomizer.uniform(.7, 1.4) for _ in phases]
            durations = [row["duration_s"] * w / sum(weights) for w in weights]
            elapsed = 0.
            for phase, duration in zip(phases, durations):
                elapsed += duration
                events.append(tagged(trial_id=row["trial_id"], policy=policy, phase=phase,
                    virtual_time_s=round(elapsed, 6), phase_duration_s=round(duration, 6),
                    status="fixture_abort" if phase == phases[-1] and not row["success"] else "fixture_transition"))
    summary = tagged(disclaimer=DISCLAIMER, groups=group_summary(rows), episodes=len(rows),
        drift_groups=[dict(policy=policy, eligible_trials=len(values), mean_grasp_drift_cm=statistics.fmean(values))
                      for policy in policies for values in [[r["grasp_drift_cm"] for r in rows
                          if r["policy"] == policy and r["grasp_drift_cm"] is not None]]],
        interpretation="31/48 and 36/48 are prescribed fictional counts. The repository's actual CEM evaluation remains 37/48 and 43/48.")
    return {"trials.jsonl": rows, "phase_events.jsonl": events}, summary


def generate(output, suite, seed=20261019):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a fresh directory; existing examples are never overwritten")
    if suite not in {"so101", "g1"} or type(seed) is not int or seed < 0:
        raise ValueError("Invalid suite or seed")
    files, summary = so101_logs(seed) if suite == "so101" else g1_logs(seed)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in files.items():
        write_jsonl(output / name, rows)
    write_json(output / "summary.json", summary)
    human = ["[SIMULATED / 模拟] " + DISCLAIMER]
    for name, rows in files.items():
        human.append(f"[SIMULATED / 模拟] file={name} records={len(rows)}")
        for row in rows[:12]:
            fields = {k: v for k, v in row.items() if k in {"policy", "mode", "step", "trial_id", "success", "failure_type", "duration_s", "blocked_ms", "action_loss", "validation_action_loss", "phase", "virtual_time_s", "disposition"}}
            human.append("[SIMULATED / 模拟] " + json.dumps(fields, ensure_ascii=False))
    with (output / "synthetic-run.log").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(human) + "\n")
    manifest = tagged(schema_version=1, suite=suite, seed=seed, disclaimer=DISCLAIMER,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reproduction_command=f"python scripts/generate_synthetic_logs.py --suite {suite} --seed {seed} --output-dir NEW_DIRECTORY",
        outcome_selection="Rank correlated fictional difficulty scores and select a prescribed number of successes; not inference outputs.",
        numeric_scope="Counts are integers. Fractions and durations derive from records. Selected means and counts are calibrated by construction.",
        files={name: dict(records=len(rows), sha256=hashlib.sha256((output / name).read_bytes()).hexdigest()) for name, rows in files.items()},
        summary_sha256=hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
        source_data_used=[], historical_evidence=False)
    write_json(output / "manifest.json", manifest)
    readme = f"""# 模拟日志示例（{suite}）

**{DISCLAIMER}**

本目录由确定性脚本生成。它只演示日志格式、数据管线和汇总；不是物理仿真输出、模型训练日志或真机记录。每条JSONL都有source=synthetic和synthetic=true，人类可读日志每行带模拟前缀。

数值按简历的约数设定，成功次数通过构造取得，部分均值经缩放校准。小数并不增加真实性。summary.json完全从本目录逐条记录计算；manifest.json记录配方、种子、脚本与文件散列。禁止将本目录混入正式评测结果。

复现：`{manifest['reproduction_command']}`。使用新目录，已有文件不会覆盖。

SO-101中的replan_events是规划线程等待事件，不是全部控制tick；blocked_ms与端到端action_age_ms分开，不能凭这些示例宣称30Hz硬实时。training.jsonl是虚构损失曲线，未运行对应训练；其中360/140是配方设定，没有对应数据集。学习曲线阈值交点也是配方，不是测得的样本效率。未生成700次真机Rollout或采集证明。

G1示例的31/48→36/48对应64.5833%→75.0%，75.0%由整数分子分母决定，不能任意添加小数。它与仓库真实CEM控制仿真37/48→43/48无关，未执行GR00T。抓持漂移只平均有值的样本，并报告有效数量。
"""
    with (output / "README.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(readme)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("so101", "g1"))
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = generate(args.output_dir, args.suite, args.seed)
        print(json.dumps(dict(source=SOURCE, suite=args.suite, output=str(args.output_dir), simulated=True), ensure_ascii=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Synthetic fixture generation failed: {exc}\n")


if __name__ == "__main__":
    main()
