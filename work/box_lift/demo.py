"""One native, frame-limited window, with an optional visual-only robot shell."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import glfw
import mujoco
import numpy as np
if __package__:
    from .task import BoxLift, ROOT, BASELINE, PARAM_NAMES
else:
    from task import BoxLift, ROOT, BASELINE, PARAM_NAMES


class Demo:
    root = ROOT
    env_factory = BoxLift
    baseline_params = BASELINE
    parameter_names = PARAM_NAMES
    first_seed = 1000
    window_title = "G1 - Learn to Lift a Box | CPU / 30 FPS"
    appearance_file = None
    appearance_manifest = None
    sidebar_width = 0
    camera_lookat = [0.20, 0, 0.88]
    camera_distance = 1.85
    camera_azimuth = 135

    def __init__(self):
        self.env = self.env_factory()
        self.learned = self.baseline_params.copy()
        self.policy = None
        self.load_policy()
        self.baseline = False
        self.paused = False
        self.seed = self.first_seed
        self.episode_count = 0
        self.success_count = 0
        self.done_since = None
        self.train_process = None
        self.train_log = None
        self.progress = {}
        self.next_status = 0
        self.train_feedback = ""
        self.drag_pos = None
        self.fps = 30.0
        self.window = None
        self.reset()
        self.render_model, self.render_data = self.env.model, self.env.data
        if self.appearance_file is not None:
            if self.appearance_manifest is not None:
                manifest = json.loads(self.appearance_manifest.read_text(encoding="utf-8"))
                physics_digest = hashlib.sha256((self.root / "scene.mjb").read_bytes()).hexdigest()
                appearance_digest = hashlib.sha256(self.appearance_file.read_bytes()).hexdigest()
                if manifest["physics_sha256"] != physics_digest or manifest["appearance_sha256"] != appearance_digest:
                    raise RuntimeError("Appearance is outdated. Run build_appearance.py for the current task.")
            self.render_model = mujoco.MjModel.from_binary_path(str(self.appearance_file))
            if (self.render_model.nq, self.render_model.nbody) != (self.env.model.nq, self.env.model.nbody):
                raise RuntimeError("Appearance and physics models have different joint/body layouts")
            self.render_data = mujoco.MjData(self.render_model)

    def load_policy(self):
        file = self.root / "policy.json"
        if file.exists():
            policy = json.loads(file.read_text(encoding="utf-8"))
            digest = hashlib.sha256((self.root / "scene.mjb").read_bytes()).hexdigest()
            if policy["model_sha256"] != digest:
                raise RuntimeError("Scene changed. Run train.py again before using its policy.")
            self.learned = np.array([policy["parameters"][name] for name in self.parameter_names])
            self.policy = policy

    def reset(self):
        self.env.reset(self.baseline_params if self.baseline else self.learned, seed=self.seed)
        self.done_since = None

    def key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        bindings = {glfw.KEY_ESCAPE: "close", glfw.KEY_P: "pause", glfw.KEY_SPACE: "pause",
                    glfw.KEY_R: "replay", glfw.KEY_N: "next", glfw.KEY_B: "compare", glfw.KEY_T: "train",
                    glfw.KEY_F5: "replay", glfw.KEY_F6: "compare", glfw.KEY_F7: "train",
                    glfw.KEY_F8: "pause", glfw.KEY_F9: "next"}
        if key in bindings:
            self.action(bindings[key])

    def action(self, action):
        if action == "close":
            glfw.set_window_should_close(self.window, True)
        elif action == "pause":
            self.paused = not self.paused
        elif action == "replay":
            self.reset()
        elif action == "next":
            self.seed += 1
            self.reset()
        elif action == "compare":
            self.baseline = not self.baseline
            self.episode_count = self.success_count = 0
            self.reset()
        elif action == "train" and self.train_process is None:
            self.paused = True
            self.train_feedback = ""
            self.train_log = (self.root / "training-live.log").open("w", encoding="utf-8")
            flags = (subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
            self.train_process = subprocess.Popen([sys.executable, str(self.root / "train.py")],
                                                 cwd=self.root, stdout=self.train_log,
                                                 stderr=subprocess.STDOUT, creationflags=flags)

    def button_rects(self, width):
        margin, gap, count = 12, 8, 6
        cell = max(1, (width - 2 * margin - (count - 1) * gap) // count)
        return [mujoco.MjrRect(margin + i * (cell + gap), 12, cell, 48) for i in range(count)]

    def mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        x, y = glfw.get_cursor_pos(window)
        ww, wh = glfw.get_window_size(window)
        fw, fh = glfw.get_framebuffer_size(window)
        x, y = x * fw / max(ww, 1), fh - y * fh / max(wh, 1)
        for rect, name in zip(self.button_rects(fw), ["pause", "replay", "next", "compare", "train", "close"]):
            if rect.left <= x <= rect.left + rect.width and rect.bottom <= y <= rect.bottom + rect.height:
                self.action(name)
                return

    def mouse(self, window, x, y):
        if self.drag_pos is None:
            self.drag_pos = (x, y)
            return
        dx, dy = x - self.drag_pos[0], y - self.drag_pos[1]
        self.drag_pos = (x, y)
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or right):
            return
        h = max(glfw.get_window_size(window)[1], 1)
        action = mujoco.mjtMouse.mjMOUSE_MOVE_V if right else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        mujoco.mjv_moveCamera(self.env.model, action, dx / h, dy / h, self.scene, self.camera)

    def scroll(self, window, x, y):
        mujoco.mjv_moveCamera(self.env.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                             0, -0.05 * y, self.scene, self.camera)

    def status(self, now):
        if now < self.next_status:
            return
        self.next_status = now + 0.5
        if self.train_process is not None:
            try:
                self.progress = json.loads((self.root / "progress.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            code = self.train_process.poll()
            if code is not None:
                self.train_process = None
                self.train_log.close()
                self.train_log = None
                if code == 0:
                    self.load_policy()
                    self.baseline = False
                    self.reset()
                    self.paused = False
                    self.train_feedback = "Training finished: policy loaded"
                else:
                    self.train_feedback = "Training failed: see training-live.log"
        e = self.env
        temp = self.root / "demo-status.tmp"
        temp.write_text(json.dumps(dict(pid=os.getpid(), phase=e.phase, fps=round(self.fps, 1),
                        paused=self.paused, training=self.train_process is not None,
                        seed=self.seed, lift_cm=round((e.data.qpos[e.box_qadr + 2] - e.initial_z) * 100, 1),
                        dual_contact=all(e.palm_contacts), **self.task_telemetry(),
                        episodes=self.episode_count, successes=self.success_count,
                        mode="baseline" if self.baseline else "learned")), encoding="utf-8")
        temp.replace(self.root / "demo-status.json")

    def task_telemetry(self):
        return dict(hold_s=round(self.env.hold, 2))

    def task_lines(self):
        e = self.env
        title = "BASELINE (untrained)" if self.baseline or self.policy is None else "LEARNED POLICY (CEM)"
        lift = (e.data.qpos[e.box_qadr + 2] - e.initial_z) * 100
        contacts = "BOTH" if all(e.palm_contacts) else " / ".join("YES" if c else "NO" for c in e.palm_contacts)
        return ["G1  |  FIXED-BASE TWO-ARM BOX LIFT", title,
                "Goal: lift >= 10 cm and hold >= 3 s",
                f"Stage: {e.phase}   {'PAUSED' if self.paused else 'PLAYING'}",
                f"Box: {e.mass:.2f} kg   Lift: {lift:.1f} cm",
                f"Palm contacts: {contacts}   Hold: {e.hold:.1f} / 3.0 s",
                f"Episodes: {self.success_count}/{self.episode_count} successful   FPS: {self.fps:.0f}"]

    def button_labels(self):
        return ["Resume [F8]" if self.paused else "Pause [F8]", "Replay [F5]", "Next [F9]",
                "Compare [F6]", "Training..." if self.train_process else "Train [F7]", "Close [Esc]"]

    def training_lines(self):
        p = self.progress
        return [f"TRAINING: generation {p.get('generation', 0)}/{p.get('generations', 5)}  "
                f"trial {p.get('candidate', 0)}/{p.get('population', 12)}"]

    def stop_worker(self):
        if self.train_process is None:
            return
        if self.train_process.poll() is None:
            if os.name == "nt":
                # A Windows venv launcher can own a second Python process.
                # Stop this owned tree, so closing the window stops its worker.
                subprocess.run(["taskkill.exe", "/PID", str(self.train_process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.train_process.terminate()
        self.train_process.wait(timeout=5)

    def run(self):
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        try:
            glfw.window_hint(glfw.SAMPLES, 0)
            self.window = glfw.create_window(1100, 760, self.window_title, None, None)
            if self.window is None:
                raise RuntimeError("Could not create the OpenGL window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.scene = mujoco.MjvScene(self.render_model, maxgeom=128)
            self.context = mujoco.MjrContext(self.render_model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self.camera = mujoco.MjvCamera()
            self.camera.lookat[:] = self.camera_lookat
            self.camera.distance = self.camera_distance
            self.camera.azimuth = self.camera_azimuth
            self.camera.elevation = -15
            self.options = mujoco.MjvOption()
            if self.appearance_file is not None:
                self.options.geomgroup[3] = 0
            self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            glfw.set_key_callback(self.window, self.key)
            glfw.set_mouse_button_callback(self.window, self.mouse_button)
            glfw.set_cursor_pos_callback(self.window, self.mouse)
            glfw.set_scroll_callback(self.window, self.scroll)
            last = time.perf_counter()
            accumulator = 0.0
            while not glfw.window_should_close(self.window):
                frame_start = time.perf_counter()
                delta = min(frame_start - last, 0.10)
                last = frame_start
                self.fps = 0.9 * self.fps + 0.1 / max(delta, 0.0001)
                glfw.poll_events()
                self.status(frame_start)
                if not self.paused and self.train_process is None:
                    if self.done_since is None:
                        accumulator += delta
                        while accumulator >= self.env.model.opt.timestep:
                            if not self.env.step():
                                self.done_since = frame_start
                                self.episode_count += 1
                                self.success_count += self.env.result()["success"]
                                break
                            accumulator -= self.env.model.opt.timestep
                    elif frame_start - self.done_since > 2.0:
                        self.seed += 1
                        self.reset()
                        accumulator = 0.0
                else:
                    accumulator = 0.0
                mujoco.mj_forward(self.env.model, self.env.data)
                if self.render_data is not self.env.data:
                    self.render_data.qpos[:] = self.env.data.qpos
                    mujoco.mj_kinematics(self.render_model, self.render_data)
                    mujoco.mj_camlight(self.render_model, self.render_data)
                width, height = glfw.get_framebuffer_size(self.window)
                if width > 0 and height > 0:
                    viewport = mujoco.MjrRect(0, 0, width, height)
                    sidebar = min(self.sidebar_width, max(0, width - 400))
                    render_viewport = mujoco.MjrRect(sidebar, 70, width - sidebar, max(1, height - 70)) if sidebar else viewport
                    if sidebar:
                        mujoco.mjr_rectangle(viewport, 0.035, 0.055, 0.075, 1.0)
                    mujoco.mjv_updateScene(self.render_model, self.render_data, self.options, None,
                                          self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                    mujoco.mjr_render(render_viewport, self.scene, self.context)
                    e = self.env
                    lines = self.task_lines()
                    if self.policy and not self.baseline:
                        lines.append(f"Unseen validation: {self.policy['validation_successes']}/{self.policy['validation_episodes']}")
                    if self.train_process is not None:
                        lines.extend(self.training_lines())
                    elif self.train_feedback:
                        lines.append(self.train_feedback)
                    if self.done_since is not None:
                        lines.append("SUCCESS - real friction contacts" if e.result()["success"] else "TRY AGAIN - goal not reached")
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                       viewport, "\n".join(lines), "", self.context)
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                       mujoco.MjrRect(0, 70, width, max(1, height - 70)),
                                       "Click buttons below. Mouse: rotate / pan / zoom", "", self.context)
                    labels = self.button_labels()
                    for rect, label in zip(self.button_rects(width), labels):
                        mujoco.mjr_label(rect, mujoco.mjtFont.mjFONT_NORMAL, label,
                                         0.16, 0.27, 0.36, 1.0, 0.94, 0.97, 1.0, self.context)
                    glfw.swap_buffers(self.window)
                time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - frame_start)))
        finally:
            if self.train_process is not None:
                self.stop_worker()
                self.train_log.close()
            if hasattr(self, "context"):
                self.context.free()
            if hasattr(self, "scene"):
                self.scene.free()
            glfw.terminate()
            (self.root / "demo-status.json").unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        Demo().run()
    except KeyboardInterrupt:
        pass
