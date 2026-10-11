"""Reuse the 30 FPS native viewer for the full pick-and-place task."""
from task import PickPlace, ROOT, BASELINE, PARAM_NAMES
from work.box_lift.demo import Demo as LiftDemo


class Demo(LiftDemo):
    root = ROOT
    env_factory = PickPlace
    baseline_params = BASELINE
    parameter_names = PARAM_NAMES
    first_seed = 3000
    window_title = "Unitree G1 - Pick and Place | 30 FPS"
    appearance_file = ROOT / "appearance.mjb"
    appearance_manifest = ROOT / "appearance-manifest.json"
    sidebar_width = 350
    camera_lookat = [0.16, 0, 0.74]
    camera_distance = 2.60
    camera_azimuth = 150

    def task_telemetry(self):
        e = self.env
        return dict(hold_s=round(e.placed_hold, 2), target_error_cm=round(e.goal_distance * 100, 2),
                    table_contact=e.table_contact, upright=round(e.upright, 3))

    def task_lines(self):
        e = self.env
        title = "BASELINE (untrained)" if self.baseline or self.policy is None else "LEARNED POLICY (CEM)"
        contacts = "BOTH" if all(e.palm_contacts) else " / ".join("YES" if c else "NO" for c in e.palm_contacts)
        lift = (e.data.qpos[e.box_qadr + 2] - e.initial_z) * 100
        return ["UNITREE G1", "PICK - CARRY - PLACE", title,
                "Goal: lift >= 10 cm",
                "Place on GREEN, then release",
                "Stable >= 1 s, error < 3.5 cm",
                f"Stage: {e.phase}   {'PAUSED' if self.paused else 'PLAYING'}",
                f"Box: {e.mass:.2f} kg",
                f"Lift: {lift:.1f} cm",
                f"Palm contacts: {contacts}",
                f"Target error: {e.goal_distance * 100:.1f} cm",
                f"Settled: {e.placed_hold:.1f} / 1.0 s",
                f"Successes: {self.success_count}/{self.episode_count}",
                f"FPS: {self.fps:.0f}"]


if __name__ == "__main__":
    Demo().run()
