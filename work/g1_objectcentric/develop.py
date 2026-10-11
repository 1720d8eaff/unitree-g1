"""Development-only grasp-height sweep; never uses frozen evaluation scenes."""
from task import ObjectCentricTask, PARAMS, ROOT
from work.g1_objectcentric.scenarios import development
import json


if __name__ == "__main__":
    e = ObjectCentricTask()
    rows = []
    for dz in (-.005, 0., .005, .010, .015):
        p = PARAMS.copy()
        p[1] = dz
        for mode in ("raw", "feedback"):
            e.reset(params=p, mode=mode, seed=9, scenario=development()[1])
            while e.step():
                pass
            r = dict(grasp_z=dz, **e.result())
            rows.append(r)
            print(dz, mode, r["success"], round(100*r["target_error_m"], 2), r["placed_hold_s"], flush=True)
    (ROOT/"grasp-height-development.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
