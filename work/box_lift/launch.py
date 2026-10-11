"""Quiet launch, log exceptions, and allow only one task window at a time."""
from pathlib import Path
import ctypes
from ctypes import wintypes
import hashlib
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parent


def main(root=ROOT, demo_factory=None):
    mutex = None
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateMutexW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        mutex = kernel.CreateMutexW(None, False, "Local\\G1BoxLift_" + hashlib.sha256(str(root).encode()).hexdigest()[:16])
        if not mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            kernel.CloseHandle(mutex)
            return
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)  # Below normal.
    try:
        with (root / "demo.log").open("w", encoding="utf-8", buffering=1) as log:
            sys.stdout = sys.stderr = log
            try:
                if demo_factory is None:
                    from demo import Demo
                    demo_factory = Demo
                demo_factory().run()
            except Exception:
                traceback.print_exc()
                if os.name == "nt":
                    ctypes.windll.user32.MessageBoxW(None, f"Could not start. See:\n{root / 'demo.log'}", "G1 Robot Task", 0x10)
    finally:
        if mutex:
            kernel.CloseHandle(mutex)


if __name__ == "__main__":
    main()
