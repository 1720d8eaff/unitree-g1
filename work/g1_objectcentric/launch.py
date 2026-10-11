from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from work.box_lift.launch import main
from work.g1_objectcentric.demo import Demo, ROOT

if __name__ == "__main__":
    main(root=ROOT, demo_factory=Demo)
