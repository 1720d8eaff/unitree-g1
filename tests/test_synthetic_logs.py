import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_fixture_generator", ROOT / "scripts/generate_synthetic_logs.py")
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class SyntheticLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.so_files, cls.so_summary = GEN.so101_logs(20261019)
        cls.g1_files, cls.g1_summary = GEN.g1_logs(20261019)

    def test_every_record_carries_machine_readable_source(self):
        for files in (self.so_files, self.g1_files):
            for name, rows in files.items():
                self.assertTrue(rows, name)
                for row in rows:
                    self.assertEqual(row["source"], "synthetic")
                    self.assertIs(row["synthetic"], True)
                    self.assertEqual(row["clock"], "virtual_monotonic")

    def test_success_rates_are_recomputed_from_integer_counts(self):
        for files, key, summary in ((self.so_files, "policy_trials.jsonl", self.so_summary["policy_groups"]),
                                    (self.so_files, "async_trials.jsonl", self.so_summary["async_groups"]),
                                    (self.g1_files, "trials.jsonl", self.g1_summary["groups"])):
            for group in summary:
                rows = [r for r in files[key] if r["condition"] == group["condition"] and r["policy"] == group["policy"]]
                successes = sum(r["success"] for r in rows)
                self.assertEqual(group["successes"], successes)
                self.assertEqual(group["trials"], len(rows))
                self.assertEqual(group["success_pct"], 100 * successes / len(rows))

    def test_prescribed_rounded_targets_and_recovery_denominator(self):
        groups = {(g["condition"], g["policy"]): g for g in self.so_summary["policy_groups"]}
        self.assertEqual(round(groups["fixed", "original"]["success_pct"]), 64)
        self.assertEqual(round(groups["fixed", "failure_aware"]["success_pct"]), 78)
        self.assertEqual(round(groups["perturbed", "original"]["success_pct"]), 37)
        self.assertEqual(round(groups["perturbed", "failure_aware"]["success_pct"]), 57)
        recovery = self.so_summary["recovery"]
        self.assertEqual((recovery["autonomous_recoveries"], recovery["recognized_failures"]), (33, 54))
        self.assertEqual(round(recovery["recovery_pct"]), 61)
        g1 = {g["policy"]: g for g in self.g1_summary["groups"]}
        self.assertEqual((g1["raw"]["successes"], g1["feedback"]["successes"]), (31, 36))
        self.assertEqual(round(g1["raw"]["success_pct"], 1), 64.6)
        self.assertEqual(g1["feedback"]["success_pct"], 75.)

    def test_long_tail_stale_actions_and_monotonic_virtual_clocks(self):
        rows = self.so_files["replan_events.jsonl"]
        for mode in ("sync", "naive_async", "aligned_async"):
            group = [r for r in rows if r["mode"] == mode]
            self.assertEqual(len({r["request_id"] for r in group}), len(group))
            stamps = [r["virtual_capture_ns"] for r in group]
            self.assertTrue(all(a < b for a, b in zip(stamps, stamps[1:])))
            self.assertGreater(max(r["blocked_ms"] for r in group), 1.3 * min(r["blocked_ms"] for r in group))
            self.assertTrue(all(r["action_age_ms"] >= r["blocked_ms"] for r in group))
        self.assertTrue(any(r["disposition"] == "discarded_stale" for r in rows))

    def test_recovery_never_happens_without_detection(self):
        for row in self.so_files["policy_trials.jsonl"]:
            self.assertFalse(row["autonomous_recovery"] and not row["recognized_failure"])

    def test_generation_is_reproducible_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            GEN.generate(first, "g1", 42)
            GEN.generate(second, "g1", 42)
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes(), path.name)
            with self.assertRaises(ValueError):
                GEN.generate(first, "g1", 42)

    def test_manifest_hashes_and_committed_fixture_sources(self):
        folder = ROOT / "examples/synthetic-resume-logs"
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        self.assertIs(manifest["historical_evidence"], False)
        self.assertEqual(manifest["source_data_used"], [])
        self.assertEqual(hashlib.sha256((ROOT / "scripts/generate_synthetic_logs.py").read_bytes()).hexdigest(), manifest["generator_sha256"])
        for name, item in manifest["files"].items():
            data = (folder / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            self.assertEqual(len(data.decode().splitlines()), item["records"])
            self.assertTrue(all(json.loads(line)["source"] == "synthetic" for line in data.decode().splitlines()))
        self.assertTrue(all(line.startswith("[SIMULATED / 模拟]") for line in (folder / "synthetic-run.log").read_text(encoding="utf-8").splitlines()))


if __name__ == "__main__":
    unittest.main()
