@echo off
echo === KHOI DONG 3D SHOE RECONSTRUCTION ===
echo.

:: Chuyen den thu muc chua file bat
cd /d "%~dp0"

:: Xoa moi truong ao cu neu Python khac 3.11
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe --version 2>&1 | findstr "3.11" >nul
    if errorlevel 1 (
        echo Phat hien .venv cu. Dang xoa va tao lai voi Python 3.11...
        rmdir /s /q .venv
    )
)

:: Tao moi truong ao neu chua co
if not exist ".venv" (
    echo Dang tao moi truong ao voi Python 3.11...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo LOI: Khong the tao moi truong ao!
        echo Vui long kiem tra Python 3.11 da duoc cai dat chua.
        pause
        exit /b 1
    )
    echo Moi truong ao da duoc tao thanh cong!
    echo.
)

:: Kich hoat moi truong ao
echo Dang kiem tra thu vien...
call .venv\Scripts\activate

:: Kiem tra xem cac thu vien cot loi da duoc cai chua
python -c "import torch, streamlit, rembg, onnxruntime, open3d, mcubes, trimesh" 2>nul
if not errorlevel 1 (
    echo Tat ca thu vien da duoc cai dat. Bo qua qua trinh cap nhat.
    goto run_app
)

:: Luon cap nhat thu vien moi lan chay neu thieu
echo Phat hien thieu thu vien, dang cap nhat pip va PyTorch...
python -m pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet

echo Dang cap nhat cac thu vien khac...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo LOI: Khong the cai dat thu vien!
    pause
    exit /b 1
)

:run_app
:: Chay ung dung Streamlit
echo Dang khoi dong Giao dien Web...
streamlit run demo/app.py

echo.
echo === DA DONG UNG DUNG ===
pause > nul
