@echo off
cd /d "%~dp0.."
echo Running GPU XY Hysteresis Sweep...
echo Parameters: T=0.89 h_max=3.0 steps=300 equil=500 meas=200
xy_hysteresis_gpu.exe 0.89 3.0 300 500 200
pause
