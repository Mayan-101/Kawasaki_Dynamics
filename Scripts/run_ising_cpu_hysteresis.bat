@echo off
cd /d "%~dp0.."
echo Running CPU Ising Hysteresis Sweep...
echo Parameters: T=1.5 h_max=3.0 steps=300 equil=500 meas=200
ising_hysteresis.exe 1.5 3.0 300 500 200
pause
