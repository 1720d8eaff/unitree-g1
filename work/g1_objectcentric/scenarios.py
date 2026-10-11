"""Frozen paired 24-scene evaluation suite, separate from development scenes."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Scenario:
    scene_id: int = -1
    start_x: float = .28
    start_y: float = -.035
    target_x: float = .28
    target_y: float = .125
    yaw_deg: float = 0.
    size_scale: float = 1.
    mass_kg: float = .4
    pose_delay_s: float = .06
    action_delay_s: float = .04
    differential_noise_m: float = .009
    common_bias_m: float = .004
    dropout_start_s: float = 3.0
    dropout_duration_s: float = .16


def suite():
    # 3 initial yaws x 4 target placements x 2 box sizes = 24 scenes.
    out = []
    for yaw in (-5., 0., 5.):
        for tx, ty in ((.265, .105), (.285, .105), (.265, .145), (.285, .145)):
            for size in (.97, 1.03):
                i = len(out)
                out.append(Scenario(i, .28 + (.004 if i % 2 else -.004), -.035,
                                    tx, ty, yaw, size, .35 + .03 * (i % 5)))
    return out


def development():
    return [Scenario(-i-1, target_x=.272 + .008*i, target_y=.115 + .005*i,
                     yaw_deg=-3 + 3*i, size_scale=.98 + .02*i) for i in range(3)]


def serialize(scenes):
    return [asdict(s) for s in scenes]


def episode_seed(scene, repeat):
    return 170000 + (scene.scene_id + 10) * 10 + repeat
