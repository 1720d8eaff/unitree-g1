"""Numerical checks for frame conventions, projection, and physical controls."""
import unittest
import numpy as np
from geometry import pose, inverse, exp_so3, log_so3, project_bimanual, relative_pose_delta, decode_relative


class GeometryTests(unittest.TestCase):
    def test_rotation_round_trip_including_pi(self):
        for angle in (0., 1e-9, .8, np.pi-1e-8, np.pi):
            axis = np.array([1., 2., -3.]) / np.sqrt(14)
            r = exp_so3(angle*axis)
            np.testing.assert_allclose(exp_so3(log_so3(r)), r, atol=2e-7)

    def test_relative_delta_is_body_frame(self):
        a = pose([.3, .2, .4], exp_so3([.2, -.4, .8]))
        b = pose([.5, -.1, .6], exp_so3([-.3, .7, -.2]))
        np.testing.assert_allclose(decode_relative(a, relative_pose_delta(a, b)), b, atol=1e-9)

    def test_shared_grasp_and_global_frame_equivariance(self):
        grasps = np.array([pose([0, .1, 0]), pose([0, -.1, 0], exp_so3([0, 0, .1]))])
        objects = [pose([.3, .0, .9], exp_so3([.04, .03, 0])), pose([.31, .01, .91], exp_so3([-.01, .01, .04]))]
        hands = np.array([o @ g for o, g in zip(objects, grasps)])
        projected, shared = project_bimanual(hands, grasps, (.7, .3))
        for h, g in zip(projected, grasps):
            np.testing.assert_allclose(h @ inverse(g), shared, atol=1e-9)
        change = pose([1., 2., 3.], exp_so3([.2, .4, -.3]))
        transformed, _ = project_bimanual([change @ h for h in hands], grasps, (.7, .3))
        np.testing.assert_allclose(transformed, [change @ h for h in projected], atol=1e-8)

    def test_independent_approach_is_unchanged(self):
        hands = np.array([pose([.2, .2, .8]), pose([.3, -.3, .9])])
        projected, shared = project_bimanual(hands, [np.eye(4)]*2, enabled=False)
        np.testing.assert_array_equal(projected, hands)
        self.assertIsNone(shared)

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            project_bimanual([np.eye(4)]*2, [np.eye(4)]*2, (-1, 1))


if __name__ == "__main__":
    unittest.main()
