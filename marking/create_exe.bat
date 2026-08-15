@echo off
setlocal
pushd "%~dp0"
"%~dp0venv\Scripts\python.exe" --version >nul 2>&1
if errorlevel 1 (
    echo Virtual environment is missing or broken.
    echo Recreate venv and install requirements.txt before building.
    set "BUILD_EXIT_CODE=1"
    goto finish
)
"%~dp0venv\Scripts\python.exe" -m PyInstaller --clean --noconfirm marking.spec
set "BUILD_EXIT_CODE=%ERRORLEVEL%"
if "%BUILD_EXIT_CODE%"=="0" (
    echo.
    echo Build completed successfully.
    if exist "%SystemRoot%\System32\ie4uinit.exe" (
        "%SystemRoot%\System32\ie4uinit.exe" -show >nul 2>&1
    )
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\refresh_exe_icon.ps1" "%~dp0dist\marking\marking.exe"
    echo Windows icon cache refresh requested.
) else (
    echo.
    echo Build failed with exit code %BUILD_EXIT_CODE%.
)

:finish
popd
pause
exit /b %BUILD_EXIT_CODE%
