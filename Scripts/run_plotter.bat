@echo off
cd /d "%~dp0..\plotter"
echo Generating Hysteresis Plots...
venv\Scripts\python.exe plot_hysteresis.py
pause
