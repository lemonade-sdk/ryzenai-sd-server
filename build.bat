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

REM Derive the paths CMakeLists.txt expects from the ONNX Runtime root:
REM   ONNXRUNTIME_LIB_DIR : <root>\lib          (contains onnxruntime.lib + DLLs)
REM   RUNTIME_DLLS_DIR    : <root>\..\deployment (ORT + VitisAI EP + custom-ops DLLs)
REM   LIB_DIRECTORY       : <root>\..\GenAI-SD\lib (stable diffusion pipeline runtime)
set "ONNXRUNTIME_LIB_DIR=%ONNXRUNTIME_ROOT%\lib"
for %%I in ("%ONNXRUNTIME_ROOT%\..") do set "RYZENAI_ROOT=%%~fI"
set "DEPLOYMENT_DIR=%RYZENAI_ROOT%\deployment"
set "GENAI_LIB_DIR=%RYZENAI_ROOT%\GenAI-SD\lib"

echo ========================================
echo Building Ryzen AI SD Server
echo ========================================
echo ONNX Runtime root: %ONNXRUNTIME_ROOT%
echo ORT lib dir:       %ONNXRUNTIME_LIB_DIR%
echo Deployment DLLs:   %DEPLOYMENT_DIR%
echo GenAI-SD lib:      %GENAI_LIB_DIR%
echo.

REM Build the optional DLL-deployment CMake args only for paths that exist
set "EXTRA_CMAKE_ARGS="
if exist "%DEPLOYMENT_DIR%" (
    set EXTRA_CMAKE_ARGS=%EXTRA_CMAKE_ARGS% -DRUNTIME_DLLS_DIR="%DEPLOYMENT_DIR%"
) else (
    echo Warning: deployment\ folder not found at %DEPLOYMENT_DIR%
)
if exist "%GENAI_LIB_DIR%" (
    set EXTRA_CMAKE_ARGS=%EXTRA_CMAKE_ARGS% -DLIB_DIRECTORY="%GENAI_LIB_DIR%"
) else (
    echo Warning: GenAI-SD\lib\ not found at %GENAI_LIB_DIR%
)

REM Create build directory
if not exist build mkdir build
cd build

REM Remove stale CMake cache so path changes take effect
if exist CMakeCache.txt del CMakeCache.txt

REM Configure
echo [1/2] Configuring...
cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DONNXRUNTIME_LIB_DIR="%ONNXRUNTIME_LIB_DIR%"%EXTRA_CMAKE_ARGS%

if errorlevel 1 (
    echo Configuration failed!
    exit /b 1
)

REM Build (CMake POST_BUILD steps deploy all runtime DLLs next to the exe)
echo.
echo [2/2] Building Release...
cmake --build . --config Release

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

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
