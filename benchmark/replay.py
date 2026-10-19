"""Recompute fresh-run success from full telemetry, without MuJoCo or weights."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import zipfile
from .evidence import ROOT, require, strict_json
from .protocol import Criteria, SuccessEvaluator
from .runner import plan


def audit_run(directory):
    directory = Path(directory).resolve()
    read = lambda name: strict_json((directory / name).read_text(encoding="utf-8"))
    completed = read("completed.json")
    require(completed["complete"] is True, "Run is incomplete")
    require(set(completed["artifacts_sha256"]) == {"manifest.json", "episodes.jsonl", "summary.json"}, "Invalid completion inventory")
    for name, digest in completed["artifacts_sha256"].items():
        require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, f"Artifact changed: {name}")
    manifest, summary = read("manifest.json"), read("summary.json")
    require(manifest["schema_version"] == 2 and summary["schema_version"] == 2, "Wrong schema")
    require(manifest["criteria"] == asdict(Criteria()), "Success criteria changed")
    require(manifest["primary_endpoint"] == "terminal_success", "Primary endpoint changed")
    require(manifest["physics_dt_s"] == .002 and manifest["telemetry_period_s"] == .01, "Sampling protocol changed")
    source_inventory = {"benchmark/runner.py", "benchmark/protocol.py", "benchmark/evidence.py",
        "work/g1_objectcentric/task.py", "work/g1_objectcentric/actions.py", "work/g1_objectcentric/geometry.py",
        "work/g1_objectcentric/scenarios.py", "work/g1_objectcentric/evaluate.py", "work/g1_objectcentric/reference-parameters.json",
        "work/box_transfer/task.py", "work/box_transfer/scene.xml", "work/box_transfer/scene.mjb", "work/box_transfer/policy.json",
        "work/box_lift/task.py"}
    require(set(manifest["source_sha256"]) == source_inventory, "Run source inventory changed")
    for name, expected in manifest["source_sha256"].items():
        source = (ROOT / name).resolve()
        require(ROOT in source.parents, "Source path escapes repository")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == expected, f"Run source changed: {name}")
    specs = manifest["episodes"]
    require(specs, "Empty plan")
    identities = [(s["scene_id"], s["repeat"], s["mode"], s["seed"]) for s in specs]
    require(len(set(identities)) == len(specs), "Duplicate plan identity")
    scenes = list(dict.fromkeys(s["scene_id"] for s in specs))
    repeats = max(s["repeat"] for s in specs) + 1
    modes = list(dict.fromkeys(s["mode"] for s in specs))
    require(specs == plan(scenes, repeats, modes), "Plan has missing, unordered or wrong-seed episodes")
    with (directory / "episodes.jsonl").open(encoding="utf-8") as stream:
        rows = [strict_json(line) for line in stream]
    require(len(rows) == len(specs) == completed["episodes"] == summary["episodes"], "Episode count mismatch")
    require(summary["complete"] is True and set(summary["modes"]) == set(modes), "Incomplete summary")
    expected_files = set()
    use_archive = (directory / "telemetry.zip").is_file()
    require(not (use_archive and (directory / "telemetry").exists()), "Ambiguous telemetry: both directory and archive exist")
    for spec, row in zip(specs, rows):
        require(all(row[k] == v for k, v in spec.items()), "Episode does not match frozen plan")
        require(type(row["legacy_success"]) is bool and type(row["simulation_steps"]) is int, "Invalid outcome/step type")
        expected_name = f"telemetry/scene-{spec['scene_id']:02d}_repeat-{spec['repeat']}_{spec['mode']}.jsonl.gz"
        require(row["telemetry_file"] == expected_name, "Unexpected telemetry path")
        expected_files.add(Path(expected_name).name)
        if use_archive:
            with zipfile.ZipFile(directory / "telemetry.zip") as archive:
                compressed = archive.read(expected_name)
        else:
            compressed = (directory / expected_name).read_bytes()
        require(hashlib.sha256(compressed).hexdigest() == row["telemetry_sha256"], "Telemetry digest mismatch")
        evaluator = SuccessEvaluator(row["initial_z_m"], Criteria(**manifest["criteria"]))
        with gzip.open(BytesIO(compressed), "rt", encoding="utf-8") as stream:
            for line in stream:
                evaluator.update(strict_json(line))
        require(math.isclose(evaluator.last_timestamp, row["simulation_end_s"], abs_tol=1e-8), "Trace omits terminal state")
        require(0 <= row["simulation_end_s"] - row["planned_duration_s"] <= .0020001, "Episode ended early or late")
        require(math.isclose(row["simulation_steps"] * .002, row["simulation_end_s"], abs_tol=1e-8), "Physics step count mismatch")
        actual = evaluator.result()
        require(set(actual) == set(row["independent"]), "Incomplete success evidence")
        for key, value in actual.items():
            stored = row["independent"][key]
            if isinstance(value, float):
                require(isinstance(stored, (int, float)) and not isinstance(stored, bool)
                        and math.isclose(stored, value, rel_tol=0, abs_tol=1e-9), f"Replayed metric mismatch: {key}")
            else:
                require(type(stored) is type(value) and stored == value, f"Replayed outcome mismatch: {key}")
    if use_archive:
        with zipfile.ZipFile(directory / "telemetry.zip") as archive:
            names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == {"telemetry/" + name for name in expected_files}, "Archive telemetry coverage mismatch")
    else:
        require({p.name for p in (directory / "telemetry").glob("*.jsonl.gz")} == expected_files, "Telemetry coverage mismatch")
    for mode in modes:
        group = [r for r in rows if r["mode"] == mode]
        expected = dict(episodes=len(group), legacy_successes=sum(r["legacy_success"] for r in group),
            terminal_successes=sum(r["independent"]["terminal_success"] for r in group),
            ever_stable_successes=sum(r["independent"]["ever_stable_success"] for r in group))
        require(summary["modes"][mode] == expected, f"Summary mismatch: {mode}")
    return dict(passed=True, episodes=len(rows), success_recomputed_from_full_telemetry=True,
                matches_archived_v1_required=False, modes=summary["modes"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(audit_run(args.directory), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Replay audit failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
