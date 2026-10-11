"""Integration checks: rigid projection, stale data, and contact-only controls."""
from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from work.g1_objectcentric.task import ObjectCentricTask, ROOT
from work.g1_objectcentric.scenarios import Scenario
from work.g1_objectcentric.actions import ActionChunk
import mujoco


def run():
    e = ObjectCentricTask()
    checks = {}
    assert e.model.neq == 0, "Object must not be welded"
    checks["no_equality_attachment"] = True
    checks["arm_dof"] = int(len(e.arm_q))
    assert checks["arm_dof"] == 14
    tests = [dict(name="nominal_feedback", scene=Scenario()),
             dict(name="no_grip", scene=Scenario(), no_grip=True),
             dict(name="no_palm_contact", scene=Scenario(), no_contact=True),
             dict(name="stale_actions", scene=Scenario(action_delay_s=.6)),
             dict(name="long_pose_dropout", scene=Scenario(dropout_duration_s=.8))]
    for item in tests:
        item = item.copy()
        name, scene = item.pop("name"), item.pop("scene")
        e.reset(seed=9, mode="feedback", scenario=scene, **item)
        while e.step():
            pass
        result = e.result()
        checks[name] = result
        print(name, json.dumps(result), flush=True)
    assert not checks["no_grip"]["success"]
    assert not checks["no_palm_contact"]["success"]
    assert checks["stale_actions"]["stale_action_steps"] > 0
    assert checks["long_pose_dropout"]["stale_pose_steps"] > 0
    assert checks["nominal_feedback"]["commanded_inconsistency_m"] < 1e-8
    size_checks = []
    for scale in (.97, 1., 1.03):
        e.reset(seed=9, scenario=Scenario(size_scale=scale), no_contact=True)
        heights, contacts = [], []
        for i in range(1000):
            mujoco.mj_step(e.model, e.data)
            if i >= 500:
                heights.append(float(e.data.qpos[e.box_qadr+2]))
                contacts.append(any({int(c.geom1), int(c.geom2)} == {e.box_geom, e.table_geom} for c in e.data.contact))
        check = dict(scale=scale, contact_fraction=float(np.mean(contacts)), height_std_m=float(np.std(heights)),
                     height_error_m=float(abs(np.mean(heights)-e.target[2])))
        assert check["contact_fraction"] > .95 and check["height_std_m"] < 1e-5 and check["height_error_m"] < .001
        size_checks.append(check)
    checks["compiled_size_table_contact"] = size_checks
    chunk = ActionChunk(1., 1.1, .02, np.zeros((40, 2, 6)))
    assert chunk.validate(1.2) and not chunk.validate(1.5)
    chunk.relative_eef[0, 0, 0] = np.nan
    try:
        chunk.validate(1.2)
        raise AssertionError("NaN action accepted")
    except ValueError:
        checks["nonfinite_action_rejected"] = True
    checks["passed"] = True
    (ROOT/"validation.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


if __name__ == "__main__":
    run()
