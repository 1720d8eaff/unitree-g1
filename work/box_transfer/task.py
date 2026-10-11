"""Contact-only two-arm pick, transfer, release, and settle task."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
from work.box_lift.task import BoxLift

PARAM_NAMES = ["compression_m", "grasp_z_offset_m", "lift_m", "reach_x_offset_m",
               "approach_s", "close_s", "lift_s", "stiffness", "transfer_s",
               "lower_s", "place_z_offset_m", "release_s", "place_x_offset_m", "place_y_offset_m"]
LOW = np.array([0.003, 0.010, 0.12, -0.006, 0.70, 0.40, 0.80, 0.80, 1.0, 0.8, 0.00, 0.6, -0.005, -0.025])
HIGH = np.array([0.014, 0.028, 0.17, 0.006, 1.20, 0.85, 1.45, 1.65, 2.0, 1.6, 0.03, 1.1, 0.055, 0.025])
BASELINE = np.array([0.008, 0.020, 0.150, 0.0, 0.9, 0.55, 1.0, 1.20, 1.5, 1.15, 0.015, 0.80, 0.0, 0.0])


class PickPlace(BoxLift):
    PHASES = ["REACH", "GRIP", "LIFT", "CARRY", "LOWER", "RELEASE", "RETREAT", "SETTLE"]

    def __init__(self):
        super().__init__(ROOT / "scene.mjb")
        self.table_geom = self.model.geom("table").id
        self.target = np.array([0.28, 0.125, 0.795])

    def ik(self, arm, position, seed):
        # The quaternion logarithm avoids the false zero of a cross-product
        # orientation error at exactly 180 degrees.
        m, d = self.model, self.ik_data
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        bounds = m.jnt_range[arm["joints"]]
        best_q, best_error = None, float("inf")
        starts = [seed, self.neutral[arm["q"]],
                  np.array([-0.4, arm["sign"] * 0.2, arm["sign"] * 0.2, 1.2, 0, 0, 0])]
        for start in starts:
            d.qpos[:] = self.neutral
            d.qpos[arm["q"]] = start
            for _ in range(220):
                mujoco.mj_forward(m, d)
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, d.site_xmat[arm["site"]])
                if quat[0] < 0:
                    quat *= -1
                norm = np.linalg.norm(quat[1:])
                ori_error = -quat[1:] * (2 * np.arctan2(norm, quat[0]) / max(norm, 1e-12))
                pos_error = position - d.site_xpos[arm["site"]]
                error = np.linalg.norm(pos_error) + 0.18 * np.linalg.norm(ori_error)
                if error < best_error:
                    best_q, best_error = d.qpos[arm["q"]].copy(), float(error)
                if error < 0.0001:
                    break
                mujoco.mj_jacSite(m, d, jp, jr, arm["site"])
                jac = np.vstack((jp[:, arm["v"]], 0.18 * jr[:, arm["v"]]))
                err = np.concatenate((pos_error, 0.18 * ori_error))
                dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 0.00004, err)
                dq *= min(1.0, 0.15 / max(np.max(np.abs(dq)), 1e-9))
                d.qpos[arm["q"]] = np.clip(d.qpos[arm["q"]] + dq, bounds[:, 0] + 0.015, bounds[:, 1] - 0.015)
            if best_error < 0.001:
                break
        self.last_ik_errors.append(best_error)
        return best_q

    def reset(self, params=BASELINE, seed=0, randomize=True, no_grip=False, no_contact=False, no_transfer=False):
        # Shared reset initializes mass, inertia, actuators and telemetry. It
        # modifies the object pose only here, before the episode starts.
        super().reset(params[:8], seed, randomize, no_grip, no_contact,
                      center=[0.28, -0.035, 0.795], prepare_trajectory=False)
        self.params = np.clip(np.asarray(params), LOW, HIGH)
        center = self.data.qpos[self.box_qadr:self.box_qadr + 3].copy()
        self.initial_z = center[2]
        compression, dz, lift, dx, approach, close, lift_time, gain, carry_time, lower_time, place_dz, release_time, place_dx, place_dy = self.params
        self.durations = [approach, close, lift_time, carry_time, lower_time, release_time, 0.9, 1.5]
        self.waypoints = []
        self.last_ik_errors = []
        dest = center.copy() if no_transfer else self.target.copy()
        dest += [place_dx, place_dy, 0]
        for arm in self.arms:
            sign = arm["sign"]
            neutral = np.array([0.20, sign * 0.22, 0.92])
            opened = center + [dx, sign * 0.145, dz]
            pinch_width = 0.145 if no_grip else 0.094 - compression
            closed = center + [dx, sign * pinch_width, dz]
            lifted = closed + [0, 0, lift]
            carried = dest + [dx, sign * pinch_width, dz + lift]
            lowered = dest + [dx, sign * pinch_width, dz + place_dz]
            released = dest + [dx, sign * 0.160, dz + place_dz]
            retreat = released + [-0.025, 0, 0.13]
            poses = []
            q = self.neutral[arm["q"]]
            for p in [neutral, opened, closed, lifted, carried, lowered, released, retreat, retreat]:
                q = self.ik(arm, p, q)
                poses.append(q)
            self.waypoints.append(np.array(poses))
            self.data.qpos[arm["q"]] = poses[0]
        self.placed_hold = self.best_placed_hold = 0.0
        self.table_contact = False
        self.goal_distance = float(np.linalg.norm(center[:2] - self.target[:2]))
        self.upright = 1.0
        self.box_speed = 0.0
        self.reward = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self

    @property
    def phase(self):
        t = self.elapsed
        for name, duration in zip(self.PHASES, self.durations):
            if t < duration:
                return name
            t -= duration
        return "DONE"

    def targets(self):
        t = self.elapsed
        phase = len(self.durations) - 1
        for i, duration in enumerate(self.durations):
            if t < duration:
                phase = i
                break
            t -= duration
        s = np.clip(t / self.durations[phase], 0, 1)
        s = s * s * (3 - 2 * s)
        return np.concatenate([poses[phase] * (1 - s) + poses[phase + 1] * s for poses in self.waypoints])

    def step(self):
        old_reward = self.reward
        running = super().step()
        self.reward = old_reward
        if self.steps % 5 == 0:
            d = self.data
            self.table_contact = any({int(c.geom1), int(c.geom2)} == {self.box_geom, self.table_geom} for c in d.contact)
            center = d.qpos[self.box_qadr:self.box_qadr + 3]
            self.goal_distance = float(np.linalg.norm(center[:2] - self.target[:2]))
            self.upright = float(d.xmat[self.box_body].reshape(3, 3)[2, 2])
            self.box_speed = float(np.linalg.norm(d.cvel[self.box_body, 3:]))
            placed = (self.table_contact and self.goal_distance < 0.035
                      and abs(center[2] - self.target[2]) < 0.02
                      and self.upright > 0.96 and self.box_speed < 0.035
                      and not any(self.palm_contacts) and self.phase in ("RETREAT", "SETTLE", "DONE"))
            self.placed_hold = self.placed_hold + 0.01 if placed else 0.0
            self.best_placed_hold = max(self.best_placed_hold, self.placed_hold)
            height = center[2] - self.initial_z
            both = all(self.palm_contacts)
            progress = np.clip(1 - self.goal_distance / 0.16, -1, 1)
            self.reward += 0.01 * (1.5 * both + 2 * min(max(height, 0) / 0.12, 1)
                                  + 5 * progress + 12 * placed)
        return running

    def result(self):
        success = bool(self.best_placed_hold >= 1.0 and self.max_height >= 0.10
                       and self.contact_time >= 0.5 and not self.fallen)
        return dict(success=success, reward=float(self.reward + 30 * self.best_placed_hold + 50 * success - 30 * self.fallen),
                    max_lift_m=float(self.max_height), placed_hold_s=round(self.best_placed_hold, 3),
                    target_error_m=float(self.goal_distance), upright=float(self.upright), speed_mps=self.box_speed,
                    dual_contact_s=round(self.contact_time, 3), mass_kg=float(self.mass),
                    max_ik_error_m=max(self.last_ik_errors), steps=self.steps)

    def episode(self, params=BASELINE, seed=0, randomize=True, **kwargs):
        self.reset(params, seed, randomize, **kwargs)
        while self.step():
            pass
        return self.result()


if __name__ == "__main__":
    import json, time
    env = PickPlace()
    start = time.perf_counter()
    env.reset(randomize=False)
    print("IK errors", np.round(env.last_ik_errors, 5))
    while env.step():
        if env.steps % 250 == 0:
            print(round(env.elapsed, 1), env.phase, env.data.qpos[env.box_qadr:env.box_qadr + 3].round(3),
                  env.palm_contacts, "upright", round(env.upright, 2), "goal", round(env.goal_distance, 3))
    print(json.dumps(env.result(), indent=2), "wall_s", time.perf_counter() - start)
