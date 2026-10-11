"""Recompile portable XML assets into a separate output directory."""
from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/recompiled"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for rel in ("box_lift/scene.xml", "box_transfer/scene.xml", "box_transfer/appearance.xml"):
        source = ROOT / "work" / rel
        model = mujoco.MjModel.from_xml_path(str(source))
        target = OUT / (rel.replace("/", "-").replace(".xml", ".mjb"))
        mujoco.mj_saveModel(model, str(target))
        print(f"{target.name}: nq={model.nq}, nu={model.nu}, nbody={model.nbody}")
