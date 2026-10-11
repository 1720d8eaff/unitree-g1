"""Verify visual poses match physics and the existing policy still succeeds."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import hashlib
import json
import time
import mujoco
import numpy as np
from task import ROOT, PickPlace, PARAM_NAMES


def validate():
    policy = json.loads((ROOT / "policy.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "appearance-manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / "scene.mjb").read_bytes()).hexdigest()
    assert digest == policy["model_sha256"] == manifest["physics_sha256"]
    visual = mujoco.MjModel.from_binary_path(str(ROOT / "appearance.mjb"))
    vd = mujoco.MjData(visual)
    env = PickPlace()
    for i in range(env.model.nbody):
        assert env.model.body(i).name == visual.body(i).name
    for i in range(env.model.njnt):
        assert env.model.joint(i).name == visual.joint(i).name
    params = [policy["parameters"][name] for name in PARAM_NAMES]
    started = time.perf_counter()
    max_position_error = max_rotation_error = 0.0
    results = []
    pose_checks = 0
    for seed in range(3000, 3020):
        env.reset(params, seed=seed)
        while env.step():
            if env.steps % 100 != 0:
                continue
            vd.qpos[:] = env.data.qpos
            mujoco.mj_kinematics(visual, vd)
            mujoco.mj_camlight(visual, vd)
            # Compare at the integrated state, including the free box pose.
            mujoco.mj_kinematics(env.model, env.data)
            max_position_error = max(max_position_error, float(np.max(np.abs(vd.xpos - env.data.xpos))))
            max_rotation_error = max(max_rotation_error, float(np.max(np.abs(vd.xmat - env.data.xmat))))
            pose_checks += 1
        results.append(dict(seed=seed, **env.result()))
    successes = sum(r["success"] for r in results)
    report = dict(successes=successes, episodes=len(results), pose_checks=pose_checks,
                  max_body_position_error_m=max_position_error, max_rotation_matrix_error=max_rotation_error,
                  physics_sha256=digest, appearance_sha256=manifest["appearance_sha256"],
                  mean_target_error_cm=float(np.mean([r["target_error_m"] for r in results])) * 100,
                  elapsed_s=round(time.perf_counter() - started, 2), results=results)
    (ROOT / "appearance-validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert max_position_error < 1e-12 and max_rotation_error < 1e-12
    assert successes == 20
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)


if __name__ == "__main__":
    validate()
