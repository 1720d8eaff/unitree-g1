"""Contact-only G1 manipulation with synthetic action and pose disturbances.

No VLA weights are loaded. The action source is the existing CEM trajectory
converted to end-effector poses. MuJoCo truth is used only by a labeled sensor
emulator and the evaluator, never presented as RGB-D visual tracking.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import sys
import json
import xml.etree.ElementTree as ET
from collections import deque
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT))
from work.box_lift.task import BoxLift
from work.box_transfer.task import PickPlace, PARAM_NAMES
from work.g1_objectcentric.geometry import pose, inverse, exp_so3, log_so3, project_bimanual, blend_pose, relative_pose_delta
from work.g1_objectcentric.actions import ActionChunk
from work.g1_objectcentric.scenarios import Scenario, suite

POLICY = json.loads((ROOT.parent / "box_transfer" / "policy.json").read_text(encoding="utf-8"))
REFERENCE = json.loads((ROOT / "reference-parameters.json").read_text(encoding="utf-8"))
PARAMS = np.array([REFERENCE["parameters"][n] for n in PARAM_NAMES])
MODES = ("raw", "smooth", "project", "feedback")


class ObjectCentricTask(PickPlace):
    control_dt = .02

    def __init__(self):
        self.yaw = 0.
        super().__init__()
        self.fk_data = mujoco.MjData(self.model)
        self.mode = "feedback"
        self.scenario = Scenario()
        self.q_bounds = np.vstack([self.model.jnt_range[a["joints"]] for a in self.arms])
        self.trace = []
        self.size_models = {1.: self.model}

    def set_box_size(self, scale):
        # Recompile size changes: editing geom_size alone leaves compiled
        # collision AABBs/rbounds stale, which can flicker tabletop contacts.
        if scale not in self.size_models:
            xml = ET.parse(ROOT.parent/"box_transfer"/"scene.xml")
            geom = xml.getroot().find(".//geom[@name='box_geom']")
            geom.set("size", " ".join(str(float(v*scale)) for v in (.065, .08, .05)))
            self.size_models[scale] = mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))
        self.model = self.size_models[scale]
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.fk_data = mujoco.MjData(self.model)

    def ik(self, arm, position, seed):
        return self.ik_pose(arm, pose(position, exp_so3([0, 0, self.yaw])), seed, 180)

    def ik_pose(self, arm, target, seed, iterations=12):
        m, d = self.model, self.ik_data
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        limits = m.jnt_range[arm["joints"]]
        starts = [seed]
        if iterations > 20:
            starts += [self.neutral[arm["q"]], np.array([-.4, arm["sign"]*.2, arm["sign"]*.2, 1.2, 0, 0, 0])]
        best_error, best_position_error, best_q = float("inf"), float("inf"), np.asarray(seed).copy()
        for start in starts:
            d.qpos[:] = self.neutral
            d.qpos[arm["q"]] = start
            for _ in range(iterations):
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                actual_r = d.site_xmat[arm["site"]].reshape(3, 3)
                pos_error = target[:3, 3] - d.site_xpos[arm["site"]]
                rot_error = log_so3(target[:3, :3] @ actual_r.T)
                score = np.linalg.norm(pos_error) + .16*np.linalg.norm(rot_error)
                if score < best_error:
                    best_error, best_position_error = score, float(np.linalg.norm(pos_error))
                    best_q = d.qpos[arm["q"]].copy()
                if score < .0001:
                    break
                mujoco.mj_jacSite(m, d, jp, jr, arm["site"])
                jac = np.vstack((jp[:, arm["v"]], .16 * jr[:, arm["v"]]))
                error = np.r_[pos_error, .16 * rot_error]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 4e-5, error)
                dq *= min(1., .15 / max(np.max(np.abs(dq)), 1e-9))
                d.qpos[arm["q"]] = np.clip(d.qpos[arm["q"]] + dq, limits[:, 0] + .015, limits[:, 1] - .015)
            if best_error < .001:
                break
        if iterations > 20:
            self.last_ik_errors.append(best_position_error)
        return best_q

    def fk(self, q):
        d = self.fk_data
        d.qpos[:] = self.neutral
        d.qpos[self.arm_q] = q
        mujoco.mj_kinematics(self.model, d)
        return np.array([pose(d.site_xpos[a["site"]], d.site_xmat[a["site"]].reshape(3, 3)) for a in self.arms])

    def object_pose(self):
        q = self.data.qpos[self.box_qadr:self.box_qadr + 7]
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, q[3:])
        return pose(q[:3], rot.reshape(3, 3))

    def reset(self, params=PARAMS, seed=170000, randomize=False, mode=None, scenario=None,
              no_grip=False, no_contact=False, no_projection=False, no_feedback=False):
        self.mode = mode or self.mode
        if self.mode not in MODES:
            raise ValueError(self.mode)
        self.scenario = scenario or self.scenario
        s = self.scenario
        self.set_box_size(s.size_scale)
        self.yaw = np.deg2rad(s.yaw_deg)
        rot = exp_so3([0, 0, self.yaw])
        self.target = np.array([s.target_x, s.target_y, .745 + .05 * s.size_scale])
        center = np.array([s.start_x, s.start_y, self.target[2]])
        BoxLift.reset(self, params[:8], seed, False, no_grip, no_contact,
                      center=center, prepare_trajectory=False)
        self.params = np.asarray(params).copy()
        self.mass = s.mass_kg
        self.model.body_mass[self.box_body] = self.mass
        x, y, z = self.model.geom_size[self.box_geom]
        self.model.body_inertia[self.box_body] = self.mass / 3 * np.array([y*y+z*z, x*x+z*z, x*x+y*y])
        mujoco.mj_setConst(self.model, self.ik_data)
        self.data.qpos[self.box_qadr+3:self.box_qadr+7] = [np.cos(self.yaw/2), 0, 0, np.sin(self.yaw/2)]
        compression, dz, lift, dx, approach, close, lift_time, gain, carry, lower, place_dz, release, place_dx, place_dy = self.params
        self.durations = [approach, close, lift_time, carry, lower, release, .9, 1.5]
        self.waypoints = []
        self.last_ik_errors = []
        for arm in self.arms:
            sign = arm["sign"]
            width = .08 * s.size_scale + .014 - compression
            if no_grip:
                width = .145
            local = lambda point: rot @ np.array(point)
            opened = center + local([dx, sign*.145, dz])
            closed = center + local([dx, sign*width, dz])
            lifted = closed + [0, 0, lift]
            carried = self.target + local([dx+place_dx, sign*width+place_dy, dz+lift])
            lowered = self.target + local([dx+place_dx, sign*width+place_dy, dz+place_dz])
            released = self.target + local([dx+place_dx, sign*.160+place_dy, dz+place_dz])
            retreat = released + [-.025, 0, .13]
            q = self.neutral[arm["q"]].copy()
            poses = []
            for p in [np.array([.20, sign*.22, .92]), opened, closed, lifted, carried, lowered, released, retreat, retreat]:
                q = self.ik(arm, p, q)
                poses.append(q)
            self.waypoints.append(np.array(poses))
            self.data.qpos[arm["q"]] = poses[0]
        self.placed_hold = self.best_placed_hold = 0.
        self.table_contact = False
        self.goal_distance = float(np.linalg.norm(center[:2]-self.target[:2]))
        self.upright = 1.
        self.box_speed = 0.
        self.reward = 0.
        self.initial_object = pose(center, rot)
        closed_hands = self.fk(np.r_[self.waypoints[0][2], self.waypoints[1][2]])
        self.grasps = np.array([inverse(self.initial_object) @ h for h in closed_hands])
        self.command_q = self.data.qpos[self.arm_q].copy()
        self.previous_hands = self.fk(self.command_q)
        self.next_control = 0.
        self.sensor_history = deque(maxlen=256)
        self.latest_observation = None
        self.latched = False
        self.grasp_calibrated = False
        self.grasp_object_origin = self.initial_object.copy()
        self.no_projection, self.no_feedback = no_projection, no_feedback
        self.rng = np.random.default_rng(seed)
        self.common_bias = self.rng.normal(size=3) * s.common_bias_m
        self.common_bias[2] *= .3
        self.sensor_bias = self.rng.normal(size=3) * .0005
        self.trace = []
        self.drift_samples = []
        self.carry_grasp_reference = None
        self.pre_residuals, self.post_residuals, self.tracking_errors = [], [], []
        self.ik_residuals = []
        self.projected_steps = self.stale_steps = self.dropout_steps = 0
        self.limit_steps = self.collision_steps = 0
        self.max_tracking = 0.
        self.feedback_offset = np.zeros(3)
        self.release_maps = np.array([np.eye(4), np.eye(4)])
        self.chunk_stamp = -1.
        self.chunk_number = -1
        self.chunk_index = 0
        self.stale_action_steps = self.collision_guard_steps = 0
        self.sensor_history.append((0., self.object_pose()))
        mujoco.mj_forward(self.model, self.data)
        return self

    def reference_q(self, timestamp):
        t = np.clip(timestamp, 0, self.duration-1e-8)
        for i, duration in enumerate(self.durations):
            if t < duration:
                fraction = t / duration
                fraction = fraction * fraction * (3-2*fraction)
                return np.r_[*[q[i]*(1-fraction)+q[i+1]*fraction for q in self.waypoints]]
            t -= duration
        return np.r_[self.waypoints[0][-1], self.waypoints[1][-1]]

    def observe(self):
        t = self.elapsed
        self.sensor_history.append((t, self.object_pose()))
        cutoff = t - self.scenario.pose_delay_s
        if self.scenario.dropout_start_s <= t < self.scenario.dropout_start_s + self.scenario.dropout_duration_s:
            self.dropout_steps += 1
            return self.latest_observation
        available = [entry for entry in self.sensor_history if entry[0] <= cutoff + 1e-9]
        if available:
            timestamp, measured = available[-1]
            measured = measured.copy()
            measured[:3, 3] += self.sensor_bias + self.rng.normal(0, .0005, 3)
            self.latest_observation = (timestamp, measured)
        return self.latest_observation

    def candidate_hands(self):
        t = self.elapsed
        number = int(t / .20 + 1e-7)
        if number != self.chunk_number:
            self.chunk_number = number
            self.chunk_stamp = number * .20 - self.scenario.action_delay_s
            # This synthetic source predicts 40 poses at 50 Hz, replanned every
            # 10 controls. Each delta would be relative to the same observation.
            self.chunk = np.array([self.fk(self.reference_q(max(0, self.chunk_stamp) + k*self.control_dt)) for k in range(40)])
            anchor = self.fk(self.reference_q(max(0, self.chunk_stamp)))
            relative = np.array([[relative_pose_delta(a, h) for a, h in zip(anchor, row)] for row in self.chunk])
            self.action_chunk = ActionChunk(max(0, self.chunk_stamp), t, self.control_dt, relative)
            self.chunk = self.action_chunk.decode(anchor)
        self.chunk_index = int(np.clip(round((t-max(0, self.chunk_stamp))/self.control_dt), 0, 39))
        hands = self.chunk[self.chunk_index].copy()
        active = self.phase in ("LIFT", "CARRY", "LOWER")
        if active:
            amp = self.scenario.differential_noise_m
            wave = .5 + .5*np.sin(5*t)
            differential = amp*np.array([.25*np.sin(4*t), .6+.4*wave, .3*np.sin(3*t)])
            for i, h in enumerate(hands):
                h[:3, 3] += self.common_bias + (1 if i == 0 else -1)*differential
                h[:3, :3] = exp_so3((1 if i == 0 else -1)*np.array([.035*np.sin(t), .025*np.cos(t), .015])) @ h[:3, :3]
        return hands

    def targets(self):
        if self.elapsed + 1e-9 < self.next_control:
            return self.command_q
        self.next_control = self.elapsed + self.control_dt
        observation = self.observe()
        candidates = self.candidate_hands()
        if self.mode == "feedback" and not self.action_chunk.validate(self.elapsed):
            self.stale_action_steps += 1
            return self.command_q
        # A grasp must be physically established before the carry constraint.
        self.latched |= all(self.palm_contacts) and self.phase in ("GRIP", "LIFT")
        if self.phase == "LIFT" and self.latched and not self.grasp_calibrated and observation is not None:
            # Calibrate at the established grasp, using the delayed pose sensor
            # and commanded palm frames (including commanded compression).
            _, grasp_object = observation
            grip_hands = self.fk(np.r_[self.waypoints[0][2], self.waypoints[1][2]])
            self.grasps = np.array([inverse(grasp_object) @ h for h in grip_hands])
            self.grasp_object_origin = grasp_object.copy()
            self.grasp_calibrated = True
        gated = self.latched and self.phase in ("LIFT", "CARRY", "LOWER")
        implied = [h @ inverse(g) for h, g in zip(candidates, self.grasps)]
        pre = float(np.linalg.norm(implied[0][:3, 3]-implied[1][:3, 3]))
        if self.mode == "smooth":
            candidates = np.array([blend_pose(a, b, .65) for a, b in zip(self.previous_hands, candidates)])
        shared = None
        if self.mode in ("project", "feedback") and not self.no_projection:
            candidates, shared = project_bimanual(candidates, self.grasps, enabled=gated)
            self.projected_steps += int(gated)
        if self.mode == "feedback" and not self.no_feedback and gated and shared is not None:
            age = float("inf") if observation is None else self.elapsed-observation[0]
            if age <= .18:
                # Compare delayed measurement with the reference at its own
                # timestamp. Use the SAME common correction for both hands.
                stamp, measured = observation
                reference_hands = self.fk(self.reference_q(stamp))
                reference_obj = sum((h @ inverse(g))[:3, 3] for h, g in zip(reference_hands, self.grasps)) / 2
                correction = .35*(reference_obj-measured[:3, 3])
                correction = np.clip(correction, -.012, .012)
                self.feedback_offset = .7*self.feedback_offset + .3*correction
            else:
                self.stale_steps += 1
                self.feedback_offset *= .90
            shared[:3, 3] += self.feedback_offset
            candidates = np.array([shared @ g for g in self.grasps])
        if self.mode in ("project", "feedback") and self.phase == "LOWER" and gated:
            nominal = self.fk(self.reference_q(self.elapsed))
            self.release_maps = np.array([pose(h[:3, 3]-n[:3, 3]) for h, n in zip(candidates, nominal)])
        elif self.mode in ("project", "feedback") and self.phase in ("RELEASE", "RETREAT", "SETTLE"):
            # Keep the corrected release origin continuous, while allowing each
            # hand to open independently. Fade only after moving away from box.
            retreat_start = sum(self.durations[:6])
            keep = 1-np.clip((self.elapsed-retreat_start)/self.durations[6], 0, 1)
            candidates = np.array([blend_pose(np.eye(4), mapping, keep) @ h
                                   for mapping, h in zip(self.release_maps, candidates)])
        self.previous_hands = candidates.copy()
        post_objects = [h @ inverse(g) for h, g in zip(candidates, self.grasps)]
        post = float(np.linalg.norm(post_objects[0][:3, 3]-post_objects[1][:3, 3]))
        # IK is seeded from the nominal path, rather than integrating drift.
        reference = self.reference_q(self.elapsed)
        command = np.r_[*[self.ik_pose(arm, target, seed, 8)
                          for arm, target, seed in zip(self.arms, candidates, np.split(reference, 2))]]
        clipped = np.clip(command, self.q_bounds[:, 0]+.015, self.q_bounds[:, 1]-.015)
        delta = np.clip(clipped-self.command_q, -2.5*self.control_dt, 2.5*self.control_dt)
        self.limit_steps += int(np.max(np.abs(command-self.command_q)) > 2.5*self.control_dt + 1e-6)
        self.command_q += delta
        # Check palm-palm clearance on candidate kinematics. This limited proxy
        # guard is explicitly not a full-body collision certificate.
        commanded_hands = self.fk(self.command_q)
        distance = mujoco.mj_geomDistance(self.model, self.fk_data, self.arms[0]["geom"], self.arms[1]["geom"], .10, None)
        if distance < .002:
            self.collision_guard_steps += 1
            self.command_q -= delta
        actual_hands = self.fk(self.data.qpos[self.arm_q])
        tracking = np.mean(np.linalg.norm(actual_hands[:, :3, 3]-candidates[:, :3, 3], axis=1))
        self.max_tracking = max(self.max_tracking, float(tracking))
        self.tracking_errors.append(float(tracking))
        if self.phase == "CARRY" and all(self.palm_contacts):
            actual_obj = self.object_pose()
            offsets = [inverse(actual_obj) @ h for h in actual_hands]
            if self.carry_grasp_reference is None:
                self.carry_grasp_reference = np.array(offsets)
            self.drift_samples.append(float(np.mean([np.linalg.norm(a[:3, 3]-g[:3, 3]) for a, g in zip(offsets, self.carry_grasp_reference)])))
        if gated:
            self.pre_residuals.append(pre)
            self.post_residuals.append(post)
            command_objects = [h @ inverse(g) for h, g in zip(commanded_hands, self.grasps)]
            self.ik_residuals.append(float(np.linalg.norm(command_objects[0][:3, 3]-command_objects[1][:3, 3])))
        if self.steps % 10 == 0:
            self.trace.append(dict(t=round(self.elapsed, 6), phase=self.phase, q=self.data.qpos[self.arm_q].tolist(),
                         command=self.command_q.tolist(), object=self.object_pose().tolist(),
                         candidates=candidates.tolist(), pre_m=pre, post_m=post,
                         contacts=list(map(bool, self.palm_contacts)), obs_timestamp=None if observation is None else observation[0]))
        return self.command_q

    def step(self):
        running = super().step()
        if self.steps % 10 == 0:
            robot_ids = {int(self.model.geom_bodyid[a["geom"]]) for a in self.arms}
            # Diagnose penetration against robot geometry, excluding box/table.
            for c in self.data.contact:
                a, b = int(c.geom1), int(c.geom2)
                bodies = {int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])}
                if c.dist < -.003 and bodies & robot_ids and self.box_geom not in (a, b) and self.table_geom not in (a, b):
                    self.collision_steps += 1
                    break
        return running

    def result(self):
        result = super().result()
        mean = lambda values: float(np.mean(values)) if values else None
        if result["success"]:
            failure = "none"
        elif self.contact_time < .5:
            failure = "grasp_not_established"
        elif self.fallen or self.max_height < .10:
            failure = "slip_or_insufficient_lift"
        elif self.goal_distance >= .035:
            failure = "placement_position"
        else:
            failure = "release_or_settle"
        result.update(mode=self.mode, scene_id=self.scenario.scene_id, failure=failure,
                      grasp_drift_m=mean(self.drift_samples), drift_sample_count=len(self.drift_samples),
                      candidate_inconsistency_m=mean(self.pre_residuals), commanded_inconsistency_m=mean(self.post_residuals),
                      ik_command_inconsistency_m=mean(self.ik_residuals),
                      mean_tracking_m=mean(self.tracking_errors), max_tracking_m=self.max_tracking,
                      projected_steps=self.projected_steps, stale_pose_steps=self.stale_steps,
                      dropout_steps=self.dropout_steps, velocity_limit_steps=self.limit_steps,
                      self_collision_diagnostic_steps=self.collision_steps)
        result.update(stale_action_steps=self.stale_action_steps, palm_collision_guard_steps=self.collision_guard_steps)
        return result


if __name__ == "__main__":
    import time
    e = ObjectCentricTask()
    for mode in MODES:
        start = time.perf_counter()
        e.reset(mode=mode, scenario=Scenario(), seed=5)
        while e.step():
            pass
        print(json.dumps(e.result()), "wall", round(time.perf_counter()-start, 2), flush=True)
