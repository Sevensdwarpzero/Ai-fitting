@echo off
cd /d "%~dp0"
echo Running AI fitting with the project Python environment...
".venv\Scripts\python.exe" pose_test.py
echo.
echo If it worked, check output_fit.jpg in this folder.
pause
