"""Run a labelled software fixture through policy -> controller -> memory transport."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .policy import NOTICE, Observation, SyntheticPolicy
from .sdk2 import ArmController, MemoryTransport


def run(output, seed=20261019, steps=75):
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("Invalid fixture step count")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    now = [0.]
    clock = lambda: now[0]
    driver = MemoryTransport(np.zeros(14), clock)
    controller = ArmController(driver, lower=[-1]*14, upper=[1]*14, max_speed=[.3]*14,
                               kp=[0]*14, kd=[0]*14, clock=clock)
    policy, rows = SyntheticPolicy(seed=seed), []
    try:
        for tick in range(steps):
            now[0] = tick * .02
            if tick % 5 == 0:
                observation = Observation(now[0], driver.q, np.zeros((2, 6)),
                    {name: np.zeros((16, 16, 3), dtype=np.uint8) for name in ("front", "side")},
                    "Synthetic interface test only", "synthetic")
                plan = policy.predict(observation)
                controller.submit(plan.actions[0], observation.timestamp, source=plan.source,
                                  observation_source=observation.source)
            status = controller.tick()
            rows.append(dict(source="synthetic", synthetic=True, tick=tick,
                             virtual_time_s=now[0], status=status, joints=driver.q.tolist()))
    finally:
        controller.close()
    trace = output / "synthetic-control.jsonl"
    trace.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    summary = dict(source="synthetic", synthetic=True, notice=NOTICE, seed=seed,
        control_ticks=len(rows), virtual_duration_s=now[0], real_robot=False,
        gr00t_weights_loaded=False, task_success_measured=False,
        control_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# Synthetic software fixture\n\n" + NOTICE +
        "\n\nTimes use a virtual clock. Joint values are algorithm output, not robot telemetry.\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=75)
    p.add_argument("--seed", type=int, default=20261019)
    a = p.parse_args()
    print(json.dumps(run(a.output_dir, a.seed, a.steps), indent=2))
