import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dataflow.collect import fresh,train_stats


class DataflowTests(unittest.TestCase):
    def test_archived_outputs_cannot_be_used_as_collection_destination(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/"results",root/"work/g1_objectcentric/dataset/new-run"):
            with self.assertRaises(ValueError):fresh(path)
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/"new";fresh(target)
            with self.assertRaises(FileExistsError):fresh(target)

    def test_stats_count_frames_not_episodes(self):
        stats=train_stats([np.array([[1.,2.],[3.,4.]]),np.array([[5.,6.]])])
        self.assertEqual(stats["count"],[3]);self.assertEqual(stats["mean"],[3.,4.])

    @unittest.skipUnless(importlib.util.find_spec("cv2"),"optional OpenCV")
    def test_no_measurement_for_missing_or_invalid_depth(self):
        from dataflow.rgbd import track
        calibration={"front":{"K":np.eye(3),"T_world_camera_opengl":np.eye(4)}}
        blank=np.zeros((32,32,3),np.uint8)
        self.assertIsNone(track({"front":(blank,np.ones((32,32)))},calibration))
        orange=np.full((32,32,3),[255,120,0],np.uint8)
        self.assertIsNone(track({"front":(orange,np.full((32,32),np.nan))},calibration))
        with self.assertRaises(ValueError):track({"front":(orange,np.ones((31,32)))},calibration)

    @unittest.skipUnless(importlib.util.find_spec("mujoco") and importlib.util.find_spec("cv2"),"optional RGB-D simulation")
    def test_rgbd_tracker_uses_rendered_measurement(self):
        from dataflow.camera import RGBDCameras
        from dataflow.rgbd import track
        from work.g1_objectcentric.task import ObjectCentricTask
        env=ObjectCentricTask();env.reset(seed=20261019,mode="feedback")
        camera=RGBDCameras(env.model,320,240)
        try:
            measurement=track(camera.frame(env.data.qpos),camera.calibration)
            self.assertIsNotNone(measurement)
            self.assertLess(np.linalg.norm(measurement.position-env.object_pose()[:3,3]),.02)
        finally:camera.close()


if __name__=="__main__":unittest.main()
