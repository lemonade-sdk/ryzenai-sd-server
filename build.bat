@echo off
REM build.bat - Quick build script for Windows
REM Usage: build.bat [path_to_onnxruntime]

setlocal

REM Check for ONNX Runtime path
if "%~1"=="" (
    echo Error: ONNX Runtime path required
    echo Usage: build.bat "C:\path\to\onnxruntime"
    exit /b 1
)

REM Join all arguments to handle unquoted paths with spaces (e.g. C:\Program Files\...)
set "ONNXRUNTIME_ROOT=%*"
REM Remove surrounding quotes if present
set "ONNXRUNTIME_ROOT=%ONNXRUNTIME_ROOT:"=%"
REM Strip trailing backslash(es) — prevents CMake quoting issues with \"
:strip_slash
if "%ONNXRUNTIME_ROOT:~-1%"=="\" (
    set "ONNXRUNTIME_ROOT=%ONNXRUNTIME_ROOT:~0,-1%"
    goto strip_slash
)
REM Strip trailing slash too
if "%ONNXRUNTIME_ROOT:~-1%"=="/" (
    set "ONNXRUNTIME_ROOT=%ONNXRUNTIME_ROOT:~0,-1%"
    goto strip_slash
)

echo ========================================
echo Building Ryzen AI SD Server
echo ========================================
echo ONNX Runtime: %ONNXRUNTIME_ROOT%
echo.

REM Create build directory
if not exist build mkdir build
cd build

REM Remove stale CMake cache so path changes take effect
if exist CMakeCache.txt del CMakeCache.txt

REM Configure
echo [1/3] Configuring...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DONNXRUNTIME_ROOT="%ONNXRUNTIME_ROOT%"

if errorlevel 1 (
    echo Configuration failed!
    exit /b 1
)

REM Build
echo.
echo [2/3] Building Release...
cmake --build . --config Release

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

REM Copy ONNX Runtime DLLs
echo.
echo [3/3] Copying ONNX Runtime DLLs...
copy "%ONNXRUNTIME_ROOT%\lib\*.dll" bin\Release\ >nul 2>&1

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo.
echo Executable in: build\bin\Release\
echo   - ryzenai-sd-server.exe
echo.

cd ..
endlocal
