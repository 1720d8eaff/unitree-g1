"""Generate actual synthetic episode files, never reinterpret existing frames."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .policy import NOTICE


def generate(output, planned=220, accepted=174, seed=20261019):
    if any(type(n) is not int for n in (planned, accepted, seed)) or not 0 < accepted <= planned <= 10000:
        raise ValueError("Need 0 < accepted <= planned <= 10000")
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(planned, accepted, replace=False).tolist())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    rows, total = [], 0
    for episode in range(planned):
        row = dict(episode_id=f"synthetic-{episode:04d}", source="synthetic", synthetic=True,
                   accepted=episode in chosen, decision_basis="seeded fixture selection, not a quality evaluation")
        if episode in chosen:
            frames = int(rng.integers(17, 38))
            time = np.arange(frames, dtype=np.float64) * .1
            state = np.sin(time[:, None] * rng.uniform(.15, .5, 14)[None]) * .08
            state += rng.normal(0, .0005, state.shape)
            action = np.roll(state, -1, axis=0)
            action[-1] = state[-1]
            rel = f"episodes/{row['episode_id']}.npz"
            np.savez_compressed(output / rel, source=np.array("synthetic"), synthetic=np.array(True),
                timestamp=time, joints=state.astype(np.float32), action=action.astype(np.float32),
                front_rgb=rng.integers(0, 256, (frames, 8, 8, 3), dtype=np.uint8),
                side_rgb=rng.integers(0, 256, (frames, 8, 8, 3), dtype=np.uint8))
            row.update(frames=frames, path=rel, sha256=hashlib.sha256((output / rel).read_bytes()).hexdigest())
            total += frames
        else:
            row.update(frames=0, exclusion_reason="synthetic fixture rejection")
        rows.append(row)
    (output / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    info = dict(source="synthetic", synthetic=True, notice=NOTICE, planned_episodes=planned,
                accepted_episodes=accepted, rejected_episodes=planned-accepted, total_frames=total,
                seed=seed, real_robot_data=False, lerobot_dataset=False,
                rgb_content="random pixels; no scene or camera calibration")
    (output / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# Synthetic collection fixture\n\n" + NOTICE +
        "\n\nContains separate generated episode files. Acceptance is a seeded fixture decision. "
        "Images are random pixels. This is not a LeRobot training dataset or physical simulation.\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--planned", type=int, default=220)
    p.add_argument("--accepted", type=int, default=174)
    p.add_argument("--seed", type=int, default=20261019)
    a = p.parse_args()
    print(json.dumps(generate(a.output_dir, a.planned, a.accepted, a.seed), indent=2))
