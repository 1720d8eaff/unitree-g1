"""Export the two-view simulation dataset; accepts --episodes."""
import runpy

if __name__ == "__main__":
    runpy.run_module("work.g1_objectcentric.export_demo", run_name="__main__")
