"""Run all 192 paired contact simulations without rendering."""
import runpy

if __name__ == "__main__":
    runpy.run_module("work.g1_objectcentric.evaluate", run_name="__main__")
