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
    -lSDL2                       ^
    -lSDL2_ttf                   ^
    -lm                          ^
    -fopenmp                     ^
    -O2                          ^
    -o xy_interactive.exe ^
    -O2 ^
    -fopenmp ^
    -std=c11 ^
    -mconsole

if %ERRORLEVEL% == 0 (
    if not exist build mkdir build
    move xy_interactive.exe build\xy_interactive.exe >nul
    echo Done: build\xy_interactive.exe
    copy xy_gpu.dll build\xy_gpu.dll >nul
    echo Copied xy_gpu.dll to build\
    copy ising.dll build\ising.dll >nul
    echo Copied ising.dll to build\
) else (
    echo FAILED
)