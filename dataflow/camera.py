"""Calibrated RGB-D rendering; image measurements never read object coordinates."""
import numpy as np


class RGBDCameras:
    def __init__(self, model, width=160, height=120):
        import mujoco
        self.mj, self.model, self.width, self.height = mujoco, model, width, height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.data = mujoco.MjData(model)
        self.calibration = {}

    def frame(self, qpos, *, depth=True):
        m = self.mj
        self.data.qpos[:] = qpos
        m.mj_forward(self.model, self.data)
        frames = {}
        for name, azimuth in (("front", 150), ("side", 75)):
            camera = m.MjvCamera()
            camera.type = m.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [.27, .04, .87]
            camera.distance, camera.azimuth, camera.elevation = 1.35, azimuth, -20
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=camera)
            self.renderer.scene.flags[m.mjtRndFlag.mjRND_SHADOW] = False
            rgb = self.renderer.render().copy()
            c = m.mjv_averageCamera(self.renderer.scene.camera[0], self.renderer.scene.camera[1])
            forward, up = np.array(c.forward), np.array(c.up)
            transform = np.eye(4)
            transform[:3, :3] = np.column_stack([np.cross(forward, up), up, -forward])
            transform[:3, 3] = c.pos
            focal = self.height / (2*np.tan(np.deg2rad(self.model.vis.global_.fovy)/2))
            intrinsics = np.array([[focal, 0, (self.width-1)/2],
                                   [0, focal, (self.height-1)/2], [0, 0, 1.]])
            self.calibration[name] = dict(K=intrinsics.tolist(), T_world_camera_opengl=transform.tolist(),
                width=self.width, height=self.height, depth_units="metres", source="simulation")
            metric_depth = None
            if depth:
                self.renderer.enable_depth_rendering()
                self.renderer.update_scene(self.data, camera=camera)
                metric_depth = self.renderer.render().copy()
                self.renderer.disable_depth_rendering()
            frames[name] = (rgb, metric_depth)
        return frames

    def close(self):
        self.renderer.close()
