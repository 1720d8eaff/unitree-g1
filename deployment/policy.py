"""GR00T N1.7 public API bridge for the repository's custom fixed-palm schema."""
from dataclasses import dataclass
import time
import numpy as np

JOINT_IDS = tuple(range(15, 29))  # G1 29DoF; left seven, then right seven.
NOTICE = "Synthetic results demonstrate software only; no trained model or real robot was evaluated."


def finite(value, shape, name):
    out = np.array(value, dtype=np.float64, copy=True)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return out


@dataclass(frozen=True)
class Observation:
    timestamp: float
    joints: np.ndarray
    eef: np.ndarray  # (2,6), absolute WORLD XYZ + rotation vector, metres/radians.
    cameras: dict
    instruction: str
    source: str

    def __post_init__(self):
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Invalid monotonic timestamp")
        if self.source not in {"synthetic", "physics_simulation", "real_robot"}:
            raise ValueError("Observation source must be explicit")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("Need a language instruction")
        for name, shape in (("joints", (14,)), ("eef", (2, 6))):
            array = finite(getattr(self, name), shape, name)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if set(self.cameras) != {"front", "side"}:
            raise ValueError("Custom schema requires front and side RGB")
        images = {}
        for name, value in self.cameras.items():
            a = np.asarray(value)
            if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3 or min(a.shape) < 1:
                raise ValueError("Cameras must be nonempty H,W,3 uint8 RGB")
            images[name] = a.copy()
            images[name].setflags(write=False)
        object.__setattr__(self, "cameras", images)

    def as_gr00t(self):
        return {"video": {k: v[None, None].copy() for k, v in self.cameras.items()},
                "state": {"left_eef": self.eef[0][None, None].astype(np.float32),
                          "right_eef": self.eef[1][None, None].astype(np.float32),
                          "arms": self.joints[None, None].astype(np.float32)},
                "language": {"annotation.human.task_description": [[self.instruction]]}}


@dataclass(frozen=True)
class JointPlan:
    observation_timestamp: float
    sample_dt: float
    actions: np.ndarray
    source: str
    inference_wall_s: float

    def __post_init__(self):
        a = np.asarray(self.actions)
        if a.ndim != 2 or a.shape[1] != 14 or not 1 <= len(a) <= 512 or not np.isfinite(a).all():
            raise ValueError("Expected finite (H,14) absolute joint radians")
        if (not np.isfinite([self.sample_dt, self.observation_timestamp, self.inference_wall_s]).all()
            or self.sample_dt <= 0 or self.observation_timestamp < 0 or self.inference_wall_s < 0):
            raise ValueError("Invalid plan timing")
        if self.source not in {"synthetic", "gr00t_local", "gr00t_remote"}:
            raise ValueError("Unknown policy provenance")
        a = np.array(a, dtype=float, copy=True)
        a.setflags(write=False)
        object.__setattr__(self, "actions", a)


class GrootBridge:
    """A model's custom modality config must match, including one-frame history.

    GR00T processors must restore `arms` to absolute joint radians. This bridge
    deliberately does not treat REAL_G1 or raw relative EEF output as compatible.
    Remote provenance means a server invocation, not verification of its weights.
    """
    def __init__(self, policy, *, sample_dt, source, model_id):
        if source not in {"gr00t_local", "gr00t_remote"} or not model_id.strip():
            raise ValueError("Need policy source and checkpoint identifier")
        if not np.isfinite(sample_dt) or sample_dt <= 0:
            raise ValueError("Need dataset action period in seconds")
        expected = {"video": {"front", "side"}, "state": {"left_eef", "right_eef", "arms"},
                    "language": {"annotation.human.task_description"},
                    "action": {"left_eef", "right_eef", "arms"}}
        configs = policy.get_modality_config()
        for name, keys in expected.items():
            config = configs[name]
            if set(config.modality_keys) != keys:
                raise ValueError(f"Checkpoint {name} keys do not match fixed-palm schema")
            if name != "action" and list(config.delta_indices) != [0]:
                raise ValueError("This adapter supplies current observations only")
        self.policy, self.sample_dt, self.source, self.model_id = policy, sample_dt, source, model_id
        self.horizon = len(configs["action"].delta_indices)
        if not 1 <= self.horizon <= 512:
            raise ValueError("Invalid checkpoint action horizon")

    def reset(self):
        self.policy.reset()

    def predict(self, observation):
        start = time.perf_counter()
        result, _ = self.policy.get_action(observation.as_gr00t())
        a = np.asarray(result["arms"])
        if a.ndim != 3 or a.shape != (1, self.horizon, 14):
            raise ValueError("GR00T must return arms with shape (1,H,14)")
        return JointPlan(observation.timestamp, self.sample_dt, a[0], self.source,
                         time.perf_counter() - start)

    @classmethod
    def local(cls, checkpoint, *, sample_dt, device="cuda:0"):
        from gr00t.policy import Gr00tPolicy
        from gr00t.data.embodiment_tags import EmbodimentTag
        policy = Gr00tPolicy(model_path=checkpoint, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
                             device=device, strict=True)
        return cls(policy, sample_dt=sample_dt, source="gr00t_local", model_id=checkpoint)

    @classmethod
    def remote(cls, host, port, *, sample_dt, model_id, timeout_ms=1000):
        from gr00t.policy.server_client import PolicyClient
        policy = PolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=False)
        if not policy.ping():
            raise ConnectionError("GR00T server did not respond")
        return cls(policy, sample_dt=sample_dt, source="gr00t_remote", model_id=model_id)


class SyntheticPolicy:
    """Small reproducible motion fixture; contains no model or task success labels."""
    source = "synthetic"
    model_id = "none: deterministic software fixture"

    def __init__(self, seed=0, horizon=20, sample_dt=.1):
        self.rng, self.horizon, self.sample_dt = np.random.default_rng(seed), horizon, sample_dt

    def reset(self):
        pass

    def predict(self, observation):
        t = np.arange(1, self.horizon + 1) * self.sample_dt
        delta = np.sin(t[:, None] * np.arange(1, 15)[None] * .2) * .015
        delta += self.rng.normal(0, .0004, delta.shape)
        return JointPlan(observation.timestamp, self.sample_dt,
                         observation.joints + delta, self.source, 0.)
