import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from benchmark.evidence import (ROOT, EvidenceError, frozen_scenes, load_rows, mode_statistics,
                                paired_statistics, strict_json, validate_design)
from benchmark.protocol import Criteria, SuccessEvaluator
from benchmark.replay import audit_run
from benchmark.runner import plan

RESULTS = ROOT / "work/g1_objectcentric/results"


class ArchivedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((RESULTS / "manifest.json").read_text())
        cls.rows = load_rows(RESULTS / "episodes.csv")

    def test_archived_design_and_rates(self):
        validate_design(self.manifest, self.rows)
        stats = mode_statistics(self.rows)
        self.assertEqual([stats[m]["successes"] for m in ("raw", "smooth", "project", "feedback")], [37, 37, 35, 43])
        self.assertTrue(all(s["episodes"] == 48 for s in stats.values()))

    def test_duplicate_missing_and_wrong_seed_fail(self):
        duplicate = copy.deepcopy(self.rows)
        duplicate[-1] = duplicate[0]
        wrong_seed = copy.deepcopy(self.rows)
        wrong_seed[0]["seed"] += 1
        for rows in (duplicate, wrong_seed, self.rows[:-1]):
            with self.subTest(rows=len(rows)), self.assertRaises(EvidenceError):
                validate_design(self.manifest, rows)

    def test_scene_mass_and_start_position_are_audited(self):
        for key in ("mass_kg", "start_x", "pose_delay_s", "target_x"):
            manifest = copy.deepcopy(self.manifest)
            manifest["scenarios"][0][key] += .01
            with self.subTest(key=key), self.assertRaises(EvidenceError):
                validate_design(manifest, self.rows)

    def test_saved_design_matches_executable_scene_generator(self):
        try:
            from work.g1_objectcentric.scenarios import suite, episode_seed
        except ImportError:
            self.skipTest("NumPy needed for original scene module")
        for expected, actual in zip(frozen_scenes(), suite()):
            self.assertEqual(expected, asdict(actual))
            self.assertEqual(episode_seed(actual, 0), 170100 + actual.scene_id * 10)

    def test_pairing_keeps_scene_repeats_together(self):
        result = paired_statistics(self.rows, "raw", "feedback", resamples=1000)
        self.assertEqual([result[k] for k in ("both_success", "baseline_only", "treatment_only", "both_failure")], [34, 3, 9, 2])
        self.assertEqual(result["success_difference_pp"], 12.5)
        self.assertEqual(result["clusters"], 24)
        self.assertLess(result["scene_bootstrap95_pp"][0], 0)
        self.assertEqual(result, paired_statistics(self.rows, "raw", "feedback", resamples=1000))

    def test_json_rejects_nan_and_duplicate_keys(self):
        for text in ('{"a":NaN}', '{"a":Infinity}', '{"a":1,"a":2}'):
            with self.subTest(text=text), self.assertRaises(EvidenceError):
                strict_json(text)


class SuccessProtocolTests(unittest.TestCase):
    def sample(self, t, *, grip=False, stable=False, z=None, **updates):
        row = dict(t=t, object_z_m=(.91 if grip else .795) if z is None else z,
                   target_z_m=.795, xy_error_m=.01 if stable else .1, upright=1., speed_mps=0.,
                   table_contact=stable, palm_contacts=[grip, grip], phase="RETREAT" if stable else "LIFT")
        row.update(updates)
        return row

    def qualified(self):
        evaluator = SuccessEvaluator(.795)
        for i in range(61):
            evaluator.update(self.sample(i / 100, grip=True))
        for i in range(61, 171):
            evaluator.update(self.sample(i / 100, stable=True))
        self.assertTrue(evaluator.result()["terminal_success"])
        return evaluator

    def test_terminal_vs_historical_stability(self):
        evaluator = self.qualified()
        result = evaluator.update(self.sample(1.71, stable=False))
        self.assertFalse(result["terminal_success"])
        self.assertTrue(result["ever_stable_success"])

    def test_contact_and_lift_required(self):
        evaluator = SuccessEvaluator(.795)
        for i in range(120):
            evaluator.update(self.sample(i / 100, stable=True))
        self.assertFalse(evaluator.result()["terminal_success"])

    def test_fall_is_irreversible(self):
        evaluator = self.qualified()
        evaluator.update(self.sample(1.71, z=.69))
        for i in range(172, 290):
            evaluator.update(self.sample(i / 100, stable=True))
        self.assertFalse(evaluator.result()["terminal_success"])
        self.assertTrue(evaluator.result()["fallen"])

    def test_gaps_reordering_and_nonfinite_are_rejected(self):
        for second in (self.sample(.02), self.sample(0.), self.sample(-.01), self.sample(.01, speed_mps=float("nan"))):
            evaluator = SuccessEvaluator(.795)
            evaluator.update(self.sample(0.))
            with self.subTest(second=second), self.assertRaises(ValueError):
                evaluator.update(second)

    def test_strict_placement_boundary(self):
        evaluator = self.qualified()
        self.assertFalse(evaluator.update(self.sample(1.71, stable=True, xy_error_m=.035))["terminal_success"])

    def test_numeric_contact_flags_rejected(self):
        with self.assertRaises(ValueError):
            SuccessEvaluator(.795).update(self.sample(0., palm_contacts=[1, 1]))


class FreshRunTests(unittest.TestCase):
    def test_plan_rejects_duplicates_and_seed_overlap(self):
        for scenes, repeats, modes in (([0, 0], 2, ["raw"]), ([0], 11, ["raw"]), ([24], 2, ["raw"]), ([0], 1, ["raw", "raw"])):
            with self.subTest(scenes=scenes), self.assertRaises(ValueError):
                plan(scenes, repeats, modes)

    def test_completion_digest_prevents_undetected_edit(self):
        original = ROOT / "results/2026-10-19"
        if not original.exists():
            self.skipTest("Fresh evaluation has not been generated")
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)
            for name in ("manifest.json", "summary.json", "episodes.jsonl", "completed.json"):
                shutil.copyfile(original / name, target / name)
            with (target / "episodes.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(EvidenceError, "Artifact changed"):
                audit_run(target)


if __name__ == "__main__":
    unittest.main()
