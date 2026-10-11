"""One lightweight native G1 window for four controller configurations."""
from pathlib import Path
import sys
import json
import hashlib
import subprocess
import os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from work.g1_objectcentric.task import ObjectCentricTask, ROOT, PARAMS, MODES, PARAM_NAMES
from work.g1_objectcentric.scenarios import Scenario, suite
from work.box_lift.demo import Demo as BaseDemo


class Demo(BaseDemo):
    root = ROOT
    env_factory = ObjectCentricTask
    baseline_params = PARAMS
    parameter_names = PARAM_NAMES
    first_seed = 70
    window_title = "Unitree G1 - Object-centric Bimanual Simulation | 30 FPS"
    appearance_file = ROOT.parent/"box_transfer"/"appearance.mjb"
    sidebar_width = 390
    camera_lookat = [.16, 0, .74]
    camera_distance = 2.60
    camera_azimuth = 150

    def __init__(self):
        self.mode_index = 3
        manifest = json.loads((ROOT.parent/"box_transfer"/"appearance-manifest.json").read_text(encoding="utf-8"))
        for kind in ("physics", "appearance"):
            file = ROOT.parent/"box_transfer"/("scene.mjb" if kind == "physics" else "appearance.mjb")
            if hashlib.sha256(file.read_bytes()).hexdigest() != manifest[kind+"_sha256"]:
                raise RuntimeError("Appearance/physics checksum mismatch")
        super().__init__()

    def load_policy(self):
        self.learned = PARAMS.copy()
        self.policy = None

    def reset(self):
        scene = Scenario() if self.seed == 70 else suite()[(self.seed-71) % 24]
        self.env.reset(PARAMS, seed=self.seed, mode=MODES[self.mode_index], scenario=scene)
        self.done_since = None
        if hasattr(self, "render_model"):
            self.render_model.geom_size[self.render_model.geom("box_geom").id] = self.env.model.geom_size[self.env.box_geom]
            self.render_model.geom_pos[self.render_model.geom("target_zone").id, :2] = self.env.target[:2]
            self.render_model.geom_pos[self.render_model.geom("start_zone").id, :2] = [scene.start_x, scene.start_y]

    def action(self, action):
        if action == "compare":
            self.mode_index = (self.mode_index+1) % len(MODES)
            self.episode_count = self.success_count = 0
            self.reset()
        elif action == "train" and self.train_process is None:
            self.paused = True
            self.train_log = (ROOT/"evaluation-live.log").open("w", encoding="utf-8")
            flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0
            self.train_process = subprocess.Popen([sys.executable, str(ROOT/"evaluate.py")], stdout=self.train_log,
                     stderr=subprocess.STDOUT, cwd=ROOT, creationflags=flags)
        else:
            super().action(action)

    def status(self, now):
        if self.train_process is not None:
            try:
                self.progress = json.loads((ROOT/"results"/"progress.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            code = self.train_process.poll()
            if code is not None:
                self.train_process = None
                self.train_log.close()
                self.train_log = None
                self.train_feedback = "192 rollouts complete: see results" if code == 0 else "Evaluation failed: see evaluation-live.log"
        super().status(now)

    def training_lines(self):
        return [f"EVALUATING {self.progress.get('done', 0)} / {self.progress.get('total', 192)}"]

    def button_labels(self):
        return ["Resume [F8]" if self.paused else "Pause [F8]", "Replay [F5]", "Scene [F9]",
                "Mode [F6]", "Evaluating..." if self.train_process else "Evaluate [F7]", "Close [Esc]"]

    def task_telemetry(self):
        e = self.env
        return dict(controller=e.mode, scene_id=e.scenario.scene_id, hold_s=round(e.placed_hold, 2),
                    target_error_cm=round(e.goal_distance*100, 2), projected_steps=e.projected_steps,
                    stale_pose_steps=e.stale_steps, stale_action_steps=e.stale_action_steps)

    def task_lines(self):
        e = self.env
        residual = e.post_residuals[-1]*100 if e.post_residuals else 0
        return ["UNITREE G1 | 14 ARM DOF", "OBJECT-CENTRIC PICK / PLACE", f"Mode: {e.mode.upper()}  [F6]",
                "Action source: CEM trajectory proxy", "Sensor: simulated delayed pose", "GR00T weights: NOT loaded",
                f"Scene: {e.scenario.scene_id}  yaw={e.scenario.yaw_deg:.0f} deg", f"Stage: {e.phase}",
                f"Box: {e.mass:.2f} kg / size x{e.scenario.size_scale:.2f}",
                f"Lift: {(e.data.qpos[e.box_qadr+2]-e.initial_z)*100:.1f} cm",
                f"Two palm contacts: {'YES' if all(e.palm_contacts) else 'NO'}",
                f"Target error: {e.goal_distance*100:.2f} cm", f"Released hold: {e.placed_hold:.1f} / 1.0 s",
                f"EEF target difference: {residual:.3f} cm", f"Projection steps: {e.projected_steps}",
                f"Stale pose / action: {e.stale_steps} / {e.stale_action_steps}",
                f"Session successes: {self.success_count}/{self.episode_count}", f"FPS: {self.fps:.0f}"]


if __name__ == "__main__":
    Demo().run()
