"""Seal every data/video artifact after an independently validated collection."""
import argparse
from pathlib import Path
from .collect import digest,write_json


def seal(root):
    root=Path(root).resolve(strict=True)
    files={p.relative_to(root).as_posix():digest(p) for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in {"SHA256.json","validation.json","validation-replay.json"}}
    write_json(root/"SHA256.json",dict(source="simulation",real_robot_data=False,files=files))
    return len(files)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args();print(seal(a.root))
