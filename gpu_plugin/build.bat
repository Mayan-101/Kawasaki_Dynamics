@echo off
setlocal

set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

if not exist %VCVARS% (
    echo ERROR: vcvarsall.bat not found at %VCVARS%
    exit /b 1
)

call %VCVARS% x64

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set OUT=..\ising_gpu.dll

echo Building ising_gpu.dll ...

"%CUDA_PATH%\bin\nvcc.exe"      ^
    -o "%OUT%"                   ^
    --shared                     ^
    ising_gpu.cu                 ^
    -arch=sm_61                  ^
    -lcurand                     ^
    -Wno-deprecated-gpu-targets  ^
    --compiler-options /O2,/MT   ^
    -Xlinker /DEFAULTLIB:Version.lib ^
    -Xlinker /DEFAULTLIB:libcmt.lib

if %ERRORLEVEL% == 0 (
    echo.
    echo Done: %OUT%
) else (
    echo.
    echo FAILED
    exit /b 1
)