"""Audit the saved paired benchmark without rerunning or rewriting it."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "work/g1_objectcentric"
RESULTS = TASK / "results"


def audit():
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        path = TASK / name if (TASK / name).is_file() else ROOT / "work/box_transfer" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"Changed source/model: {path.name}"
    recorded = json.loads((RESULTS / "audit.json").read_text(encoding="utf-8"))
    for rel, expected in recorded["shared_dependency_sha256"].items():
        assert hashlib.sha256((ROOT / rel.replace("\\", "/")).read_bytes()).hexdigest() == expected, rel
    with (RESULTS / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(int(r["scene_id"]), int(r["repeat"]), r["mode"]) for r in rows}
    expected_keys = {(s["scene_id"], repeat, mode) for s in manifest["scenarios"]
                     for repeat in range(manifest["repetitions"]) for mode in manifest["modes"]}
    assert len(rows) == len(keys) == 192 and keys == expected_keys
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    for mode, stats in summary["modes"].items():
        selected = [r for r in rows if r["mode"] == mode]
        assert len(selected) == stats["episodes"] == 48
        assert sum(r["success"].lower() in ("true", "1") for r in selected) == stats["successes"]
        mean_cm = 100 * sum(float(r["target_error_m"]) for r in selected) / len(selected)
        assert math.isclose(mean_cm, stats["mean_target_error_cm"], abs_tol=1e-10)
    traces = list((RESULTS / "traces").glob("*.jsonl.gz"))
    assert len(traces) == 192
    for sid, repeat, mode in sorted(keys):
        path = RESULTS / "traces" / f"scene-{sid:02d}_repeat-{repeat}_{mode}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            count = sum(1 for line in stream if isinstance(json.loads(line), dict))
        assert count > 0, f"Empty trace: {path.name}"
    print(json.dumps({"passed":True, "episodes":len(rows), "traces":len(traces),
                      "source_hashes_match":True, "summary_matches_csv":True}))


if __name__ == "__main__":
    audit()
