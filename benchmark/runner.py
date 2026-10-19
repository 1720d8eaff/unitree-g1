"""Run fresh paired simulations with complete success telemetry; never overwrite v1."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from .evidence import MODES, ROOT, SWITCHES, frozen_scenes, seed_for
from .protocol import Criteria, SuccessEvaluator


def plan(scene_ids, repeats, modes):
    if (not scene_ids or any(type(i) is not int or not 0 <= i < 24 for i in scene_ids)
        or len(set(scene_ids)) != len(scene_ids) or type(repeats) is not int or not 1 <= repeats <= 10
        or not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes)):
        raise ValueError("Invalid or duplicate scenes, repetitions or modes")
    return [dict(scene_id=s, repeat=r, mode=m, seed=seed_for(s, r))
            for s in scene_ids for r in range(repeats) for m in modes]


def telemetry(env):
    center = env.data.qpos[env.box_qadr:env.box_qadr + 3]
    pairs = [{int(c.geom1), int(c.geom2)} for c in env.data.contact]
    return dict(t=round(float(env.elapsed), 9), object_z_m=float(env.data.qpos[env.box_qadr + 2]),
        target_z_m=float(env.target[2]), xy_error_m=float(sum((center[i] - env.target[i]) ** 2 for i in range(2)) ** .5),
        upright=float(env.data.xmat[env.box_body].reshape(3, 3)[2, 2]),
        speed_mps=float(sum(v * v for v in env.data.cvel[env.box_body, 3:]) ** .5),
        table_contact={env.box_geom, env.table_geom} in pairs,
        palm_contacts=[{env.box_geom, arm["geom"]} in pairs for arm in env.arms], phase=env.phase)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run(output, scene_ids, repeats=2, modes=MODES):
    episodes = plan(scene_ids, repeats, modes)
    # Import simulator only when actually running, leaving replay/audit CPU-only.
    import numpy as np
    import mujoco
    from work.g1_objectcentric.scenarios import Scenario
    from work.g1_objectcentric.task import ObjectCentricTask
    from work.g1_objectcentric.evaluate import summarize, paired
    output = Path(output).resolve()
    protected = ROOT / "work/g1_objectcentric/results"
    if output == protected or protected in output.parents or output in protected.parents:
        raise ValueError("Output overlaps the archived results")
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing run: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "telemetry").mkdir()
    files = [ROOT / "benchmark" / n for n in ("runner.py", "protocol.py", "evidence.py")]
    files += [ROOT / "work/g1_objectcentric" / n for n in
              ("task.py", "actions.py", "geometry.py", "scenarios.py", "evaluate.py", "reference-parameters.json")]
    files += [ROOT / "work/box_transfer" / n for n in ("task.py", "scene.xml", "scene.mjb", "policy.json")]
    files += [ROOT / "work/box_lift/task.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    manifest = dict(schema_version=2, status="planned", suite="v1_scene_grid_fresh_telemetry",
        complete_v1_design=scene_ids == list(range(24)) and repeats == 2 and tuple(modes) == MODES,
        scope="CEM surrogate and simulated object pose; no GR00T model or real robot",
        criteria=asdict(Criteria()), duration_estimator="100Hz right-endpoint integration, first sample has dt=0",
        primary_endpoint="terminal_success", secondary_endpoints=["ever_stable_success", "legacy_success"],
        seed_formula="170000 + (scene_id + 10) * 10 + repeat", episodes=episodes,
        scenarios=[frozen_scenes()[i] for i in scene_ids], module_switches={m: SWITCHES[m] for m in modes},
        source_sha256=hashes, physics_dt_s=.002, telemetry_period_s=.01,
        git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                     platform=platform.platform(), dependencies={n: importlib.metadata.version(n) for n in ("numpy", "mujoco", "glfw")}))
    # The complete plan, code hashes and endpoint are written before any rollout.
    write_json(output / "manifest.json", manifest)
    started, rows, legacy_rows = time.perf_counter(), [], []
    env = ObjectCentricTask()
    with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
        for index, spec in enumerate(episodes, 1):
            env.reset(seed=spec["seed"], mode=spec["mode"], scenario=Scenario(**frozen_scenes()[spec["scene_id"]]))
            evaluator = SuccessEvaluator(float(env.initial_z))
            name = f"scene-{spec['scene_id']:02d}_repeat-{spec['repeat']}_{spec['mode']}.jsonl.gz"
            with gzip.open(output / "telemetry" / name, "xt", encoding="utf-8") as trace:
                sample = telemetry(env)
                evaluator.update(sample)
                trace.write(json.dumps(sample, allow_nan=False) + "\n")
                while True:
                    running = env.step()
                    if env.steps % 5 == 0 or not running:
                        sample = telemetry(env)
                        evaluator.update(sample)
                        trace.write(json.dumps(sample, allow_nan=False) + "\n")
                    if not running:
                        break
            legacy = dict(repeat=spec["repeat"], seed=spec["seed"], **env.result())
            row = dict(**spec, initial_z_m=float(env.initial_z), legacy_success=legacy["success"],
                       planned_duration_s=float(env.duration), simulation_end_s=float(env.elapsed), simulation_steps=int(env.steps),
                       independent=evaluator.result(), telemetry_file="telemetry/" + name,
                       telemetry_sha256=hashlib.sha256((output / "telemetry" / name).read_bytes()).hexdigest())
            rows.append(row)
            legacy_rows.append(legacy)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(f"{index}/{len(episodes)} scene={spec['scene_id']} repeat={spec['repeat']} {spec['mode']} "
                  f"legacy={legacy['success']} terminal={row['independent']['terminal_success']}", flush=True)
    mode_summary = {}
    for mode in modes:
        group = [r for r in rows if r["mode"] == mode]
        mode_summary[mode] = dict(episodes=len(group), legacy_successes=sum(r["legacy_success"] for r in group),
            terminal_successes=sum(r["independent"]["terminal_success"] for r in group),
            ever_stable_successes=sum(r["independent"]["ever_stable_success"] for r in group))
    summary = dict(schema_version=2, complete=True, episodes=len(rows), wall_s=time.perf_counter() - started,
                   modes=mode_summary, scope=manifest["scope"])
    if tuple(modes) == MODES:
        summary["legacy_statistics"] = summarize(legacy_rows)
    if set(modes) == set(MODES) and repeats == 2:
        summary["legacy_paired_raw_feedback"] = paired(legacy_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "completed.json", dict(complete=True, episodes=len(rows),
        artifacts_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ("manifest.json", "episodes.jsonl", "summary.json")}))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", nargs="+", type=int, default=list(range(24)))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args(argv)
    try:
        run(args.output_dir, args.scenes, args.repeats, tuple(args.modes))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
