import json
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from deployment.policy import GrootBridge, JointPlan, Observation
from deployment.sdk2 import ArmController, MemoryTransport, UnitreeTransport
from deployment.demo import run
from deployment.collection_fixture import generate


def observation():
    return Observation(1., np.zeros(14), np.zeros((2, 6)),
        {k: np.zeros((8, 8, 3), np.uint8) for k in ("front", "side")}, "test fixture", "synthetic")


class FakePolicy:
    def get_modality_config(self):
        return {k: SimpleNamespace(modality_keys=v, delta_indices=list(range(5)) if k=="action" else [0]) for k, v in {
            "video": ["front", "side"], "state": ["left_eef", "right_eef", "arms"],
            "language": ["annotation.human.task_description"], "action": ["left_eef", "right_eef", "arms"]}.items()}

    def get_action(self, obs):
        self.last = obs
        return {"arms": np.ones((1, 5, 14), dtype=np.float32)*.1}, {}

    def reset(self):
        pass


class DeploymentTests(unittest.TestCase):
    def test_model_contract_and_shapes(self):
        policy = FakePolicy()
        bridge = GrootBridge(policy, sample_dt=.1, source="gr00t_remote", model_id="fake-test-policy")
        plan = bridge.predict(observation())
        self.assertEqual(plan.actions.shape, (5, 14))
        self.assertEqual(policy.last["video"]["front"].shape, (1, 1, 8, 8, 3))
        self.assertEqual(policy.last["state"]["arms"].dtype, np.float32)
        self.assertEqual(plan.source, "gr00t_remote")

    def test_incompatible_embodiment_rejected(self):
        class Wrong(FakePolicy):
            def get_modality_config(self):
                c = super().get_modality_config()
                c["state"].modality_keys = ["real_g1"]
                return c
        with self.assertRaises(ValueError):
            GrootBridge(Wrong(), sample_dt=.1, source="gr00t_local", model_id="wrong")

    def test_nonfinite_action_rejected(self):
        with self.assertRaises(ValueError):
            JointPlan(0., .1, np.full((2, 14), np.nan), "synthetic", 0.)

    def test_schema_rejects_wrong_camera_dtype(self):
        with self.assertRaises(ValueError):
            Observation(0., np.zeros(14), np.zeros((2, 6)),
                {k: np.zeros((8, 8, 3), float) for k in ("front", "side")}, "fixture", "synthetic")

    def test_velocity_bound_and_watchdog(self):
        clock = [0.]
        t = MemoryTransport(np.zeros(14), lambda: clock[0])
        c = ArmController(t, lower=[-1]*14, upper=[1]*14, max_speed=[1]*14,
                          kp=[0]*14, kd=[0]*14, clock=lambda: clock[0])
        c.submit(np.ones(14)*.5, 0., source="synthetic", observation_source="synthetic")
        c.tick()
        clock[0] = .02
        c.tick()
        np.testing.assert_allclose(t.q, .02)
        clock[0] = .5
        self.assertEqual(c.tick(), "watchdog_release")
        self.assertEqual(t.weight, 0.)

    def test_synthetic_cannot_reach_hardware(self):
        t = MemoryTransport(np.zeros(14), lambda: 0.)
        t.synthetic = False
        c = ArmController(t, lower=[-1]*14, upper=[1]*14, max_speed=[1]*14,
                          kp=[0]*14, kd=[0]*14, clock=lambda: 0.)
        with self.assertRaises(ValueError):
            c.submit(np.zeros(14), 0., source="synthetic", observation_source="synthetic")
        with self.assertRaises(ValueError):
            c.submit(np.zeros(14), 0., source="gr00t_remote", observation_source="synthetic")
        with self.assertRaises(ValueError):
            UnitreeTransport("", enable_hardware=False)

    def test_stale_and_out_of_order_commands(self):
        clock = [1.]
        c = ArmController(MemoryTransport(np.zeros(14), lambda: clock[0]), lower=[-1]*14,
            upper=[1]*14, max_speed=[1]*14, kp=[0]*14, kd=[0]*14, clock=lambda: clock[0])
        with self.assertRaises(ValueError):
            c.submit(np.zeros(14), .5, source="synthetic", observation_source="synthetic")
        c.submit(np.zeros(14), 1., source="synthetic", observation_source="synthetic")
        with self.assertRaises(ValueError):
            c.submit(np.zeros(14), .99, source="synthetic", observation_source="synthetic")

    def test_sdk2_message_maps_only_arm_ids_and_release_weight(self):
        t = object.__new__(UnitreeTransport)
        t.command = SimpleNamespace(motor_cmd=[SimpleNamespace(q=0.,dq=0.,tau=0.,kp=0.,kd=0.) for _ in range(35)], crc=0)
        t.crc = SimpleNamespace(Crc=lambda message: 123)
        written = []
        t.publisher = SimpleNamespace(Write=lambda message: written.append(message.crc))
        t.write(np.ones(14)*.2, np.ones(14)*10, np.ones(14))
        self.assertTrue(all(t.command.motor_cmd[i].q==.2 for i in range(15,29)))
        self.assertTrue(all(t.command.motor_cmd[i].q==0. for i in range(15)))
        self.assertEqual(t.command.motor_cmd[29].q, 1.)
        t.release()
        self.assertEqual(t.command.motor_cmd[29].q, 0.)
        self.assertEqual(written, [123,123])

    def test_collection_counts_actual_files_and_frames(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"collection"
            info = generate(path, planned=5, accepted=3, seed=7)
            files = list((path/"episodes").glob("*.npz"))
            self.assertEqual(len(files), 3)
            frames = 0
            for file in files:
                with np.load(file, allow_pickle=False) as a:
                    self.assertTrue(bool(a["synthetic"]))
                    frames += len(a["timestamp"])
            self.assertEqual(info["total_frames"], frames)
            self.assertNotEqual(info["total_frames"], 3)
            self.assertTrue(all(json.loads(s)["synthetic"] for s in (path/"episodes.jsonl").read_text().splitlines()))
            with self.assertRaises(FileExistsError):
                generate(path, planned=5, accepted=3)

    def test_demo_reproducible_and_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = run(Path(d)/"a", steps=10), run(Path(d)/"b", steps=10)
            self.assertEqual(a["control_trace_sha256"], b["control_trace_sha256"])
            self.assertFalse(a["gr00t_weights_loaded"])
            self.assertFalse(a["task_success_measured"])

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "optional MuJoCo integration")
    def test_simulation_new_actions_do_not_call_cem_reference(self):
        from unittest.mock import patch
        from deployment.simulation import run as simulate
        from deployment.policy import SyntheticPolicy
        from work.g1_objectcentric.task import ObjectCentricTask
        with tempfile.TemporaryDirectory() as d:
            with patch.object(ObjectCentricTask, "reference_q", side_effect=AssertionError("CEM fallback")):
                result = simulate(Path(d)/"smoke", SyntheticPolicy(), duration_s=.02)
            self.assertEqual(result["policy_source"], "synthetic")
            self.assertFalse(result["task_completed"])
            self.assertTrue((Path(d)/"smoke/manifest.json").is_file())

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "optional MuJoCo integration")
    def test_simulation_cannot_write_into_archived_results(self):
        from deployment.simulation import run as simulate
        from deployment.policy import SyntheticPolicy
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            simulate(root/"work/g1_objectcentric/results/new-model-run", SyntheticPolicy())


if __name__ == "__main__":
    unittest.main()
