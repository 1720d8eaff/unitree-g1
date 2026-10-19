"""Independent, replayable success criteria for sampled physical telemetry."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Criteria:
    min_lift_m: float = .10
    min_dual_contact_s: float = .5
    target_xy_m: float = .035
    target_z_m: float = .02
    min_upright: float = .96
    max_speed_mps: float = .035
    stable_s: float = 1.
    fallen_z_m: float = .70
    max_sample_gap_s: float = .010001

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Criteria must be finite and positive")


class SuccessEvaluator:
    """Integrate 100Hz telemetry; gaps invalidate a run rather than count as stability.

    Contact/stability durations use right endpoint sampling. Resolution is one
    sample interval, which is recorded in the protocol. Missing data, duplicate
    times and nonfinite physical fields are errors, never successful holds.
    """
    def __init__(self, initial_z, criteria=None):
        if not math.isfinite(initial_z):
            raise ValueError("Invalid initial height")
        self.initial_z = initial_z
        self.criteria = criteria or Criteria()
        self.last_timestamp = None
        self.current_stable = self.best_stable = self.dual_contact = self.max_lift = 0.
        self.fallen = False
        self.samples = 0

    def update(self, row):
        numeric = ("t", "object_z_m", "target_z_m", "xy_error_m", "upright", "speed_mps")
        if any(isinstance(row[k], bool) or not isinstance(row[k], (int, float))
               or not math.isfinite(row[k]) for k in numeric):
            raise ValueError("Nonfinite or nonnumeric physical telemetry")
        t = row["t"]
        if t < 0 or row["xy_error_m"] < 0 or row["speed_mps"] < 0 or not -1.000001 <= row["upright"] <= 1.000001:
            raise ValueError("Physical telemetry out of range")
        contacts = row["palm_contacts"]
        if (not isinstance(contacts, list) or len(contacts) != 2
            or any(type(c) is not bool for c in contacts) or type(row["table_contact"]) is not bool):
            raise ValueError("Contact flags must be booleans")
        if row["phase"] not in {"REACH", "GRIP", "LIFT", "CARRY", "LOWER", "RELEASE", "RETREAT", "SETTLE", "DONE"}:
            raise ValueError("Unknown phase")
        dt = 0. if self.last_timestamp is None else t - self.last_timestamp
        if self.last_timestamp is not None and (dt <= 0 or dt > self.criteria.max_sample_gap_s):
            raise ValueError("Telemetry gap, duplicate or reordered sample")
        c = self.criteria
        stable = (row["table_contact"] and row["xy_error_m"] < c.target_xy_m
            and abs(row["object_z_m"] - row["target_z_m"]) < c.target_z_m
            and row["upright"] > c.min_upright and row["speed_mps"] < c.max_speed_mps
            and not any(contacts) and row["phase"] in {"RETREAT", "SETTLE", "DONE"})
        self.current_stable = self.current_stable + dt if stable else 0.
        self.best_stable = max(self.best_stable, self.current_stable)
        self.dual_contact += dt if all(contacts) else 0.
        self.max_lift = max(self.max_lift, row["object_z_m"] - self.initial_z)
        self.fallen |= row["object_z_m"] < c.fallen_z_m
        self.last_timestamp, self.samples = t, self.samples + 1
        return self.result()

    def result(self):
        c = self.criteria
        prerequisites = (self.samples > 0 and self.max_lift >= c.min_lift_m
                         and self.dual_contact + 1e-9 >= c.min_dual_contact_s and not self.fallen)
        return dict(terminal_success=bool(prerequisites and self.current_stable + 1e-9 >= c.stable_s),
                    ever_stable_success=bool(prerequisites and self.best_stable + 1e-9 >= c.stable_s),
                    current_stable_s=self.current_stable, best_stable_s=self.best_stable,
                    dual_contact_s=self.dual_contact, max_lift_m=self.max_lift,
                    fallen=bool(self.fallen), samples=self.samples)
