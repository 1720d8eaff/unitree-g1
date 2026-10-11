"""World-frame SE(3) poses and object-centric rigid grasp projection."""
import numpy as np


def skew(v):
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_so3(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    k = skew(v)
    if angle < 1e-7:
        return np.eye(3) + k + .5 * k @ k
    return np.eye(3) + np.sin(angle) / angle * k + (1 - np.cos(angle)) / angle**2 * k @ k


def log_so3(r):
    r = np.asarray(r, dtype=float)
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-7:
        return .5 * vee
    if np.pi - angle < 1e-5:
        # The symmetric part identifies the axis even when sin(angle) is zero.
        vals, vectors = np.linalg.eigh((r + r.T) / 2)
        axis = vectors[:, np.argmax(vals)]
        if np.dot(axis, vee) < 0:
            axis *= -1
        return axis * angle
    return vee * angle / (2 * np.sin(angle))


def pose(p, r=None):
    out = np.eye(4)
    out[:3, 3] = p
    out[:3, :3] = np.eye(3) if r is None else r
    return out


def inverse(t):
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def mean_pose(poses, weights=None):
    """Minimize weighted translation squares and SO(3) geodesic squares.

    This is a product-metric pose mean; rotation is in radians and position in
    metres. It is not a claim of globally solving arbitrary SE(3) optimization.
    The local Karcher iteration is intended for nearby grasp predictions.
    """
    ts = np.asarray(poses, dtype=float)
    w = np.ones(len(ts)) if weights is None else np.asarray(weights, dtype=float)
    if ts.shape[1:] != (4, 4) or len(w) != len(ts) or not np.all(np.isfinite(ts)):
        raise ValueError("Expected finite (N,4,4) poses and N weights")
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and nonzero")
    w = w / w.sum()
    r = ts[np.argmax(w), :3, :3].copy()
    for _ in range(12):
        tangent = sum(wi * log_so3(r.T @ ti[:3, :3]) for wi, ti in zip(w, ts))
        if np.linalg.norm(tangent) < 1e-10:
            break
        r = r @ exp_so3(tangent)
    return pose(np.sum(w[:, None] * ts[:, :3, 3], axis=0), r)


def project_bimanual(hands, grasps, weights=(1., 1.), enabled=True):
    """G_i = T_object_hand_i; candidate object O_i = H_i inv(G_i)."""
    hands = np.asarray(hands, dtype=float)
    if not enabled:
        return hands.copy(), None
    implied = [h @ inverse(g) for h, g in zip(hands, grasps)]
    shared = mean_pose(implied, weights)
    return np.array([shared @ g for g in grasps]), shared


def relative_pose_delta(current, target):
    """Body-frame delta: T_next = T_current @ Delta; never mix with world deltas."""
    delta = inverse(current) @ target
    return np.r_[delta[:3, 3], log_so3(delta[:3, :3])]


def decode_relative(current, delta):
    return current @ pose(np.asarray(delta)[:3], exp_so3(np.asarray(delta)[3:]))


def blend_pose(a, b, fraction):
    return pose((1 - fraction) * a[:3, 3] + fraction * b[:3, 3],
                a[:3, :3] @ exp_so3(fraction * log_so3(a[:3, :3].T @ b[:3, :3])))
