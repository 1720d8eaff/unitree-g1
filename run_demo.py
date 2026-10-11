"""Start the frame-limited native demo from the repository root."""
from work.g1_objectcentric.launch import main, Demo, ROOT

if __name__ == "__main__":
    main(root=ROOT, demo_factory=Demo)
