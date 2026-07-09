@echo off
setlocal
echo === KHOI DONG 3D SHOE RECONSTRUCTION ===
echo.

:: Chuyen den thu muc chua file bat
cd /d "%~dp0"
set "VENV_PY=%cd%\.venv\Scripts\python.exe"

powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Kiem tra moi truong Python' -PercentComplete 10"

:: Xoa moi truong ao cu neu Python khac 3.11
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe --version 2>&1 | findstr "3.11" >nul
    if errorlevel 1 (
        powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Phat hien .venv cu. Dang xoa...' -PercentComplete 20"
        rmdir /s /q .venv
    )
)

:: Tao moi truong ao neu chua co
if not exist ".venv" (
    powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Dang tao moi truong ao Python 3.11...' -PercentComplete 30"
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo LOI: Khong the tao moi truong ao!
        pause
        exit /b 1
    )
)
set "VENV_PY=%cd%\.venv\Scripts\python.exe"

powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Kich hoat moi truong ao...' -PercentComplete 40"
if not exist "%VENV_PY%" (
    echo LOI: Khong tim thay Python trong moi truong ao!
    pause
    exit /b 1
)

powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Kiem tra cac thu vien can thiet...' -PercentComplete 60"
"%VENV_PY%" -c "import torch, streamlit, rembg, onnxruntime, open3d, mcubes, trimesh" 2>nul
if not errorlevel 1 (
    powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Thu vien da hoan tat' -PercentComplete 90"
    goto run_app
)

powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Dang cai dat thu vien Pytorch...' -PercentComplete 70"
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Dang cai dat cac thu vien khac...' -PercentComplete 85"
"%VENV_PY%" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo LOI: Khong the cai dat thu vien!
    pause
    exit /b 1
)

:run_app
powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Status 'Hoan tat!' -PercentComplete 100"
:: Tat progress bar
powershell -Command "Write-Progress -Activity 'Khoi dong he thong' -Completed"

echo.
echo ==============================================
echo   HE THONG DA SAN SANG!
echo ==============================================
echo.
set PYTHONPATH=%cd%
"%VENV_PY%" -m streamlit run demo\app.py

echo.
echo === DA DONG UNG DUNG ===
pause > nul
