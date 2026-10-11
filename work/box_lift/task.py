"""A small contact-based task; only the G1 arms and free box are dynamic.

There is deliberately no weld, attachment, or scripted object motion. The box
is moved solely by MuJoCo contact forces. CEM learns trajectory parameters.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
PARAM_NAMES = ["compression_m", "grasp_z_offset_m", "lift_m", "reach_x_offset_m",
               "approach_s", "close_s", "lift_s", "stiffness"]
LOW = np.array([0.001, 0.006, 0.115, -0.012, 0.65, 0.35, 0.70, 0.65])
HIGH = np.array([0.016, 0.033, 0.170, 0.012, 1.25, 0.80, 1.50, 1.65])
BASELINE = np.array([0.008, 0.021, 0.140, 0.0, 0.9, 0.55, 1.0, 1.2])


class BoxLift:
    def __init__(self, model_path=None):
        self.model = mujoco.MjModel.from_binary_path(str(model_path or ROOT / "scene.mjb"))
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.box_body = self.model.body("box").id
        self.box_geom = self.model.geom("box_geom").id
        self.box_qadr = int(self.model.jnt_qposadr[self.model.joint("box_free").id])
        self.arms = []
        self.neutral = self.model.qpos0.copy()
        for side, sign in [("left", 1), ("right", -1)]:
            names = [f"{side}_{part}_joint" for part in
                     ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
            joints = np.array([self.model.joint(n).id for n in names])
            qadrs = self.model.jnt_qposadr[joints]
            dofs = self.model.jnt_dofadr[joints]
            motors = np.array([np.flatnonzero(self.model.actuator_trnid[:, 0] == j)[0] for j in joints])
            self.arms.append(dict(side=side, sign=sign, joints=joints, q=qadrs,
                                  v=dofs, u=motors, site=self.model.site(side + "_grasp").id,
                                  geom=self.model.geom(side + "_palm").id))
            self.neutral[qadrs] = [0.35, sign * 0.18, 0, 0.87, 0, 0, 0]
        self.arm_q = np.concatenate([a["q"] for a in self.arms])
        self.arm_v = np.concatenate([a["v"] for a in self.arms])
        self.arm_u = np.concatenate([a["u"] for a in self.arms])
        self.kp = np.tile([100, 120, 80, 100, 35, 25, 25], 2)
        self.kd = np.tile([8, 9, 6, 8, 3, 2, 2], 2)
        self.elapsed = 0.0
        self.params = BASELINE.copy()
        self.palm_contacts = [False, False]
        self.hold = 0.0
        self.best_hold = 0.0
        self.max_height = 0.0
        self.reward = 0.0
        self.last_ik_errors = []

    def ik(self, arm, position, seed):
        d, m = self.ik_data, self.model
        d.qpos[:] = self.neutral
        d.qpos[arm["q"]] = seed
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        bounds = m.jnt_range[arm["joints"]]
        for _ in range(220):
            mujoco.mj_forward(m, d)
            rot = d.site_xmat[arm["site"]].reshape(3, 3)
            ori_error = 0.5 * sum(np.cross(rot[:, i], np.eye(3)[:, i]) for i in range(3))
            pos_error = position - d.site_xpos[arm["site"]]
            if np.linalg.norm(pos_error) < 0.00008 and np.linalg.norm(ori_error) < 0.001:
                break
            mujoco.mj_jacSite(m, d, jp, jr, arm["site"])
            jac = np.vstack((jp[:, arm["v"]], 0.18 * jr[:, arm["v"]]))
            err = np.concatenate((pos_error, 0.18 * ori_error))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 0.00004, err)
            dq *= min(1.0, 0.15 / max(np.max(np.abs(dq)), 1e-9))
            d.qpos[arm["q"]] = np.clip(d.qpos[arm["q"]] + dq, bounds[:, 0] + 0.015, bounds[:, 1] - 0.015)
        mujoco.mj_forward(m, d)
        self.last_ik_errors.append(float(np.linalg.norm(position - d.site_xpos[arm["site"]])))
        return d.qpos[arm["q"]].copy()

    def reset(self, params=BASELINE, seed=0, randomize=True, no_grip=False, no_contact=False,
              center=None, prepare_trajectory=True):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.params = np.clip(np.asarray(params), LOW, HIGH)
        rng = np.random.default_rng(seed)
        center = np.array([0.35, 0.0, 0.795] if center is None else center, dtype=float)
        mass = 0.4
        if randomize:
            center[:2] += rng.uniform([-0.015, -0.01], [0.015, 0.01])
            mass = rng.uniform(0.30, 0.55)
        m.body_mass[self.box_body] = mass
        m.body_inertia[self.box_body] = mass / 3 * np.array([0.08**2 + 0.05**2, 0.065**2 + 0.05**2, 0.065**2 + 0.08**2])
        for arm in self.arms:
            m.geom_contype[arm["geom"]] = 0 if no_contact else 1
            m.geom_conaffinity[arm["geom"]] = 0 if no_contact else 1
        mujoco.mj_setConst(m, self.ik_data)
        d.qpos[:] = self.neutral
        d.qpos[self.box_qadr:self.box_qadr + 3] = center
        self.initial_z = center[2]
        self.mass = mass
        compression, dz, lift, dx, approach, close, lift_time, gain = self.params
        self.durations = [approach, close, lift_time, 3.2]
        self.waypoints = []
        self.last_ik_errors = []
        for arm in self.arms if prepare_trajectory else []:
            sign = arm["sign"]
            neutral_pos = np.array([0.20, sign * 0.22, 0.92])
            open_pos = center + [dx, sign * 0.145, dz]
            closed_pos = center + [dx, sign * (0.094 - compression), dz]
            if no_grip:
                closed_pos[1] = open_pos[1]
            lift_pos = closed_pos + [0, 0, lift]
            seed_q = self.neutral[arm["q"]]
            poses = []
            for pos in [neutral_pos, open_pos, closed_pos, lift_pos]:
                seed_q = self.ik(arm, pos, seed_q)
                poses.append(seed_q)
            self.waypoints.append(np.array(poses))
            d.qpos[arm["q"]] = poses[0]
        self.elapsed = self.hold = self.best_hold = self.max_height = self.reward = 0.0
        self.contact_time = 0.0
        self.palm_contacts = [False, False]
        self.fallen = False
        self.steps = 0
        mujoco.mj_forward(m, d)
        return self

    @property
    def duration(self):
        return sum(self.durations)

    @property
    def phase(self):
        t = self.elapsed
        for i, duration in enumerate(self.durations):
            if t < duration:
                return ["REACH", "GRIP", "LIFT", "HOLD"][i]
            t -= duration
        return "DONE"

    def targets(self):
        t = self.elapsed
        phase = 3
        for i, duration in enumerate(self.durations[:3]):
            if t < duration:
                phase = i
                break
            t -= duration
        result = []
        for poses in self.waypoints:
            if phase == 3:
                result.append(poses[3])
            else:
                s = np.clip(t / self.durations[phase], 0, 1)
                s = s * s * (3 - 2 * s)
                result.append(poses[phase] * (1 - s) + poses[phase + 1] * s)
        return np.concatenate(result)

    def step(self):
        m, d = self.model, self.data
        error = self.targets() - d.qpos[self.arm_q]
        torques = self.params[7] * self.kp * error - self.kd * d.qvel[self.arm_v] + d.qfrc_bias[self.arm_v]
        limits = m.actuator_ctrlrange[self.arm_u]
        d.ctrl[self.arm_u] = np.clip(torques, limits[:, 0], limits[:, 1])
        mujoco.mj_step(m, d)
        self.elapsed = d.time
        self.steps += 1
        # mj_step positions precede integration by one timestep; refresh only
        # every 5 steps for telemetry instead of duplicating dynamics work.
        if self.steps % 5 == 0:
            self.palm_contacts = [False, False]
            for c in d.contact:
                pair = {int(c.geom1), int(c.geom2)}
                if self.box_geom in pair:
                    for i, arm in enumerate(self.arms):
                        self.palm_contacts[i] |= arm["geom"] in pair
            height = float(d.qpos[self.box_qadr + 2] - self.initial_z)
            self.max_height = max(self.max_height, height)
            both = all(self.palm_contacts)
            if both:
                self.contact_time += 0.01
            if both and height >= 0.10:
                self.hold += 0.01
                self.best_hold = max(self.best_hold, self.hold)
            else:
                self.hold = 0.0
            self.fallen |= d.qpos[self.box_qadr + 2] < 0.70
            self.reward += 0.01 * (3 * np.clip(height / 0.14, -1, 1.2) + 0.5 * both + 2 * (both and height >= 0.10))
        return self.elapsed < self.duration

    def result(self):
        return dict(success=bool(self.best_hold >= 3.0 and not self.fallen),
                    reward=float(self.reward + 8 * self.best_hold - 15 * self.fallen),
                    max_lift_m=float(self.max_height), hold_s=round(self.best_hold, 3),
                    dual_contact_s=round(self.contact_time, 3), mass_kg=float(self.mass),
                    max_ik_error_m=max(self.last_ik_errors), steps=self.steps)

    def episode(self, params=BASELINE, seed=0, randomize=True, **kwargs):
        self.reset(params, seed, randomize, **kwargs)
        while self.step():
            pass
        return self.result()


if __name__ == "__main__":
    import json, time
    env = BoxLift()
    start = time.perf_counter()
    env.reset(randomize=False)
    for a, poses in zip(env.arms, env.waypoints):
        print(a["side"], poses.round(3).tolist())
    print("IK errors:", env.last_ik_errors)
    while env.step():
        if env.steps % 250 == 0:
            print(round(env.elapsed, 2), env.phase, env.data.qpos[env.box_qadr:env.box_qadr + 3].round(4), env.palm_contacts)
    print(json.dumps(env.result(), indent=2), "wall_s", time.perf_counter() - start)
