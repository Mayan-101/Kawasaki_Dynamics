@echo off
cd /d "%~dp0.."
echo --- Building GPU Plugins ---
cd gpu_plugin
call build.bat
cd ..

echo.
echo --- Building XY Interactive ---
call build_main.bat

echo.
echo --- Building Ising Interactive ---
C:\msys64\ucrt64\bin\gcc.exe Ising.c utils/pcg_random.c -IC:\msys64\ucrt64\include\SDL2 -LC:\msys64\ucrt64\lib -lSDL2 -lSDL2_ttf -lm -fopenmp -O2 -std=c11 -mconsole -o Ising.exe
if %ERRORLEVEL% == 0 echo Done: Ising.exe

echo.
echo --- Building Hysteresis Executables ---
C:\msys64\ucrt64\bin\gcc.exe ising_hysteresis.c utils/pcg_random.c -o ising_hysteresis.exe -O2 -fopenmp -std=c11
if %ERRORLEVEL% == 0 echo Done: ising_hysteresis.exe

C:\msys64\ucrt64\bin\gcc.exe ising_hysteresis_gpu.c ising_gpu_loader.c utils/pcg_random.c -o ising_hysteresis_gpu.exe -O2 -fopenmp -std=c11
if %ERRORLEVEL% == 0 echo Done: ising_hysteresis_gpu.exe

C:\msys64\ucrt64\bin\gcc.exe xy_hysteresis_gpu.c gpu_loader.c utils/pcg_random.c -o xy_hysteresis_gpu.exe -O2 -fopenmp -std=c11
if %ERRORLEVEL% == 0 echo Done: xy_hysteresis_gpu.exe

echo.
echo --- Build Complete ---
pause
