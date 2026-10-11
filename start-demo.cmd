@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo First create .venv and install requirements.txt. See README.md.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "run_demo.py"
