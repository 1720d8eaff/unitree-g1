"""Paired 24 x 2 x 4 physical rollouts; no target outcome is hardcoded."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys
import argparse
import hashlib
import json
import csv
import gzip
import time
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from work.g1_objectcentric.task import ObjectCentricTask, ROOT, MODES, PARAMS
from work.g1_objectcentric.scenarios import suite, development, serialize, episode_seed


def wilson(k, n):
    z = 1.96
    p = k/n
    d = 1+z*z/n
    mid = (p+z*z/(2*n))/d
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(100*(mid-half)), float(100*(mid+half))]


def summarize(rows):
    out = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        n, k = len(selected), sum(r["success"] for r in selected)
        mean = lambda key: float(np.mean([r[key] for r in selected if r[key] is not None])) if any(r[key] is not None for r in selected) else None
        out[mode] = dict(episodes=n, successes=k, success_pct=100*k/n, wilson95_pct=wilson(k, n),
                         mean_target_error_cm=100*mean("target_error_m"),
                         mean_grasp_drift_cm=100*mean("grasp_drift_m") if mean("grasp_drift_m") is not None else None,
                         drift_eligible_episodes=sum(r["drift_sample_count"] > 0 for r in selected),
                         mean_command_inconsistency_cm=100*mean("commanded_inconsistency_m") if mean("commanded_inconsistency_m") is not None else None,
                         mean_tracking_cm=100*mean("mean_tracking_m"), failures=dict(Counter(r["failure"] for r in selected)),
                         mean_ik_command_inconsistency_cm=100*mean("ik_command_inconsistency_m") if mean("ik_command_inconsistency_m") is not None else None,
                         stale_pose_steps=sum(r["stale_pose_steps"] for r in selected),
                         stale_action_steps=sum(r["stale_action_steps"] for r in selected))
    return out


def paired(rows):
    indexed = {(r["scene_id"], r["repeat"], r["mode"]): r for r in rows}
    counts = dict(both_success=0, raw_only=0, feedback_only=0, both_failure=0)
    for sid, repeat, mode in indexed:
        if mode != "raw":
            continue
        a = indexed[sid, repeat, "raw"]["success"]
        b = indexed[sid, repeat, "feedback"]["success"]
        counts["both_success" if a and b else "raw_only" if a else "feedback_only" if b else "both_failure"] += 1
    # Scene-level paired bootstrap preserves the grouping of two repeats.
    ids = sorted({r["scene_id"] for r in rows})
    differences = np.array([np.mean([100*(indexed[i, r, "feedback"]["success"]-indexed[i, r, "raw"]["success"])
                                    for r in sorted({row["repeat"] for row in rows if row["scene_id"] == i})]) for i in ids])
    rng = np.random.default_rng(817)
    boot = np.mean(rng.choice(differences, (4000, len(ids)), replace=True), axis=1)
    counts.update(success_difference_pp=float(np.mean(differences)), scene_bootstrap95_pp=np.percentile(boot, [2.5, 97.5]).tolist())
    return counts


def run(development_run=False, traces=True):
    start = time.perf_counter()
    scenes = development() if development_run else suite()
    out = ROOT / ("development-results" if development_run else "results")
    out.mkdir(exist_ok=True)
    trace_dir = out / "traces"
    if traces:
        trace_dir.mkdir(exist_ok=True)
    manifest = dict(schema_version=1, suite="development" if development_run else "frozen_evaluation",
                    action_source="CEM trajectory surrogate; NOT GR00T or a learned VLA",
                    sensing="simulated noisy delayed object pose; NOT RGB-D tracking",
                    modes=list(MODES), repetitions=2, scenarios=serialize(scenes), params=PARAMS.tolist(),
                    timestep_s=.002, pose_control_s=.02, action_horizon=40, execution_horizon=10,
                    success_criteria=dict(min_lift_m=.10, min_dual_contact_s=.5, target_xy_m=.035,
                        target_z_m=.02, min_upright=.96, max_speed_mps=.035, released_stable_s=1.),
                    files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [ROOT/"task.py", ROOT/"geometry.py", ROOT/"scenarios.py", ROOT/"actions.py", ROOT/"evaluate.py", ROOT/"reference-parameters.json", ROOT.parent/"box_transfer"/"scene.mjb", ROOT.parent/"box_transfer"/"scene.xml", ROOT.parent/"box_transfer"/"policy.json"]})
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = ObjectCentricTask()
    rows = []
    total = len(scenes)*2*len(MODES)
    with (out/"episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = None
        for scene in scenes:
            for repeat in range(2):
                seed = episode_seed(scene, repeat)
                for mode in MODES:
                    env.reset(seed=seed, mode=mode, scenario=scene)
                    while env.step():
                        pass
                    row = dict(repeat=repeat, seed=seed, **env.result())
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    file.flush()
                    if traces:
                        name = f"scene-{scene.scene_id:02d}_repeat-{repeat}_{mode}.jsonl.gz"
                        with gzip.open(trace_dir/name, "wt", encoding="utf-8") as trace:
                            for sample in env.trace:
                                trace.write(json.dumps(sample, separators=(",", ":"))+"\n")
                    progress = dict(done=len(rows), total=total, last=row, elapsed_s=round(time.perf_counter()-start, 2))
                    (out/"progress.json").write_text(json.dumps(progress), encoding="utf-8")
                    print(f"{len(rows)}/{total} {mode} scene={scene.scene_id} success={row['success']} error={100*row['target_error_m']:.2f}cm", flush=True)
    summary = dict(scope=manifest["action_source"], episodes=len(rows), wall_s=time.perf_counter()-start,
                   modes=summarize(rows), paired_raw_feedback=paired(rows))
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--no-traces", action="store_true")
    args = parser.parse_args()
    run(args.development, not args.no_traces)
