"""Reuse the lightweight G1 arms with a marked pick and place table."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import mujoco

ROOT = Path(__file__).resolve().parent
LIFT_ROOT = ROOT.parent / "box_lift"


def build():
    if not (LIFT_ROOT / "scene.xml").exists():
        sys.path.insert(0, str(LIFT_ROOT))
        from build_scene import build as build_lift
        build_lift()
    tree = ET.parse(LIFT_ROOT / "scene.xml")
    root = tree.getroot()
    root.set("model", "G1 fixed-base pick and place")
    world = root.find("worldbody")
    world.find("body[@name='box']").set("pos", "0.28 -0.035 0.795")
    world.find("geom[@name='table']").set("size", "0.23 0.33 0.025")
    world.find("geom[@name='table']").set("pos", "0.36 0 0.72")
    for name, y, color in [("start_zone", -0.035, "0.18 0.38 0.57 1"),
                           ("target_zone", 0.125, "0.15 0.65 0.32 1")]:
        ET.SubElement(world, "geom", name=name, type="box", pos=f"0.28 {y} 0.7455",
                      size="0.095 0.105 0.0005", rgba=color, contype="0", conaffinity="0")
    ET.indent(tree)
    tree.write(ROOT / "scene.xml", encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    mujoco.mj_saveModel(model, str(ROOT / "scene.mjb"))
    print(f"Pick-place model: {model.nq} qpos, {model.nu} arm motors, {model.ngeom} simple geometries")


if __name__ == "__main__":
    build()
