"""Launch one pick-and-place demo quietly and below normal CPU priority."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
from work.box_lift.launch import main


def factory():
    from demo import Demo
    return Demo()


if __name__ == "__main__":
    main(ROOT, factory)
