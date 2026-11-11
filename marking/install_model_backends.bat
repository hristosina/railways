@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON310=C:\Users\Kristina\AppData\Local\Programs\Python\Python310\python.exe"
set "VENV_PYTHON=%~dp0venv310\Scripts\python.exe"
set "RTDETR_DIR=%~dp0third_party\RT-DETR"

if /I "%~1"=="--check-script" goto :script_check

echo ============================================================
echo Installing model backends for marking
echo YOLOv8s, YOLOv12s, RT-DETR-R18, RT-DETRv2-S, RF-DETR-Nano
echo ============================================================
echo.

if not exist "%PYTHON310%" goto :python_missing

if exist "%VENV_PYTHON%" goto :venv_ready
echo Creating the separate venv310 environment...
"%PYTHON310%" -m venv "%~dp0venv310"
if errorlevel 1 goto :failed

:venv_ready
echo Updating pip...
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo.
echo Installing base dependencies and CUDA PyTorch 2.5.1...
echo The PyTorch download is about 2.45 GB and may take some time.
"%VENV_PYTHON%" -m pip install -r "%~dp0requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :failed

echo.
echo Installing RF-DETR and RT-DETR report dependencies...
"%VENV_PYTHON%" -m pip install -r "%~dp0requirements-model-backends.txt"
if errorlevel 1 goto :failed

if exist "%RTDETR_DIR%\rtdetrv2_pytorch\tools\train.py" goto :rtdetr_ready
echo.
echo Downloading the official RT-DETR repository...
if not exist "%~dp0third_party" mkdir "%~dp0third_party"
git clone --depth 1 https://github.com/lyuwenyu/RT-DETR.git "%RTDETR_DIR%"
if errorlevel 1 goto :failed

:rtdetr_ready
echo.
echo Applying RT-DETR compatibility for current TorchVision...
"%VENV_PYTHON%" "%~dp0tools\patch_rtdetr_torchvision.py"
if errorlevel 1 goto :failed

echo.
echo Checking the model environment...
"%VENV_PYTHON%" "%~dp0tools\check_model_backends.py"
if errorlevel 1 goto :failed

echo.
echo Installation completed successfully.
echo Send the final check output to Codex so integration can continue.
echo.
pause
exit /b 0

:python_missing
echo ERROR: Python 3.10 was not found at:
echo %PYTHON310%
goto :failed

:script_check
echo Installer script syntax check passed.
exit /b 0

:failed
echo.
echo INSTALLATION DID NOT COMPLETE. The relevant error is shown above.
echo Send the last lines from this window to Codex.
echo.
pause
exit /b 1
