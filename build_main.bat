@echo off
setlocal

set GCC=C:\msys64\ucrt64\bin\gcc.exe
set SDL_INC=C:\msys64\ucrt64\include\SDL2
set SDL_LIB=C:\msys64\ucrt64\lib

if not exist build mkdir build

"%GCC%"                          ^
    XY_model_gpu.c                      ^
    gpu_loader.c                 ^
    utils/pcg_random.c           ^
    -I"%SDL_INC%"                ^
    -L"%SDL_LIB%"                ^
    -o Kawasaki_Dynamics.exe ^
    -lSDL2                       ^
    -lSDL2_ttf                   ^
    -lm                          ^
    -fopenmp                     ^
    -O2                          ^
    -std=c11                     ^
    -mconsole

if %ERRORLEVEL% == 0 (
    echo Done: build\Kawasaki_Dynamics.exe
    copy ising_gpu.dll build\ising_gpu.dll >nul
    echo Copied ising_gpu.dll to build\
) else (
    echo FAILED
)