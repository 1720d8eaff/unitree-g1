"""Optional GR00T-driven MuJoCo rollout; frozen benchmark sources stay unchanged.

The simulation pauses while inference runs. This is not a real-time benchmark.
It consumes only restored absolute `arms` predictions, not CEM actions. Existing
scene reset and task-duration definitions are reused, with an independent success
evaluator. Synthetic backend is a smoke test, not a GR00T result.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .policy import GrootBridge, Observation, SyntheticPolicy


def run(output, policy, *, duration_s=1., project=False, seed=20261019):
    import mujoco
    from benchmark.runner import telemetry
    from benchmark.protocol import SuccessEvaluator
    from work.g1_objectcentric.task import ObjectCentricTask
    from work.g1_objectcentric.geometry import log_so3, project_bimanual
    from work.g1_objectcentric.scenarios import Scenario
    if not np.isfinite(duration_s) or not 0 < duration_s <= 60:
        raise ValueError("Duration must be in (0,60] simulation seconds")
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[1]
    for protected in (root / "work/g1_objectcentric/results", root / "work/g1_objectcentric/dataset", root / "results"):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Choose an output outside archived results and data")
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(dict(status="planned", seed=seed,
        observation_source="physics_simulation", policy_source=policy.source,
        synthetic=policy.source=="synthetic", real_robot=False, model_id=policy.model_id,
        requested_duration_sim_s=duration_s, project=bool(project),
        inference_blocks_simulation=True), indent=2)+"\n", encoding="utf-8")
    calls, samples = [], []

    class PolicyTask(ObjectCentricTask):
        def targets(self):
            if self.elapsed + 1e-9 < self.next_control:
                return self.command_q
            self.next_control = self.elapsed + self.control_dt
            if self.plan is None or self.elapsed >= self.replan_at:
                images = {}
                for name, azimuth in (("front", 150), ("side", 75)):
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = [.27, .04, .87]
                    camera.distance, camera.azimuth, camera.elevation = 1.35, azimuth, -20
                    self.renderer.update_scene(self.data, camera=camera)
                    images[name] = self.renderer.render().copy()
                poses = self.fk(self.data.qpos[self.arm_q])
                eef = np.array([np.r_[p[:3, 3], log_so3(p[:3, :3])] for p in poses])
                observation = Observation(self.elapsed, self.data.qpos[self.arm_q], eef, images,
                    "Use both arms to pick up the box, carry it to the target and release it.", "physics_simulation")
                self.plan = policy.predict(observation)
                self.replan_at = self.elapsed + min(.2, len(self.plan.actions) * self.plan.sample_dt)
                calls.append(dict(observation_sim_s=self.elapsed, source=self.plan.source,
                    inference_wall_s=self.plan.inference_wall_s, horizon=len(self.plan.actions)))
            index = int(np.clip((self.elapsed-self.plan.observation_timestamp)/self.plan.sample_dt,
                                0, len(self.plan.actions)-1))
            target = self.plan.actions[index]
            # Optional shared-pose projection; no CEM reference or pose feedback.
            if project and all(self.palm_contacts):
                hands, _ = project_bimanual(self.fk(target), self.grasps)
                target = np.concatenate([self.ik_pose(arm, hand, q, 30)
                    for arm, hand, q in zip(self.arms, hands, np.split(target, 2))])
            target = np.clip(target, self.q_bounds[:, 0]+.015, self.q_bounds[:, 1]-.015)
            self.command_q += np.clip(target-self.command_q, -2.5*self.control_dt, 2.5*self.control_dt)
            return self.command_q

    env = PolicyTask()
    env.reset(seed=seed, mode="raw", scenario=Scenario())
    env.plan, env.replan_at = None, 0.
    evaluator = SuccessEvaluator(float(env.initial_z))
    renderer = mujoco.Renderer(env.model, height=240, width=320)
    env.renderer = renderer
    try:
        policy.reset()
        evaluator.update(telemetry(env))
        while env.elapsed < duration_s:
            running = env.step()
            if env.steps % 5 == 0 or not running:
                sample = telemetry(env)
                evaluator.update(sample)
                samples.append(dict(**sample, joints=env.data.qpos[env.arm_q].tolist(),
                                    command=env.command_q.tolist(), observation_source="physics_simulation",
                                    policy_source=policy.source))
            if not running:
                break
    finally:
        renderer.close()
    trace = output / "telemetry.jsonl"
    trace.write_text("".join(json.dumps(x, allow_nan=False)+"\n" for x in samples), encoding="utf-8")
    result = dict(observation_source="physics_simulation", policy_source=policy.source,
        synthetic=policy.source == "synthetic", real_robot=False, model_id=policy.model_id,
        checkpoint_provenance="local execution or remote server declaration; no weight hash attestation",
        project=bool(project), duration_sim_s=float(env.elapsed), criteria=asdict(evaluator.criteria),
        task_result=evaluator.result(), task_completed=bool(env.elapsed >= env.duration-.003),
        calls=calls, inference_blocks_simulation=True, telemetry_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--policy", choices=("synthetic", "local", "remote"), default="synthetic")
    p.add_argument("--checkpoint")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--sample-dt", type=float, default=.1)
    p.add_argument("--duration", type=float, default=1.)
    p.add_argument("--project", action="store_true")
    a = p.parse_args()
    if a.policy != "synthetic" and not a.checkpoint:
        p.error("Need --checkpoint for model identity and matching custom modality configuration")
    policy = (SyntheticPolicy(sample_dt=a.sample_dt) if a.policy == "synthetic" else
              GrootBridge.local(a.checkpoint, sample_dt=a.sample_dt) if a.policy == "local" else
              GrootBridge.remote(a.host, a.port, sample_dt=a.sample_dt, model_id=a.checkpoint))
    print(json.dumps(run(a.output_dir, policy, duration_s=a.duration, project=a.project), indent=2))
