@echo off
REM === Set full path to Python 3.10 installation ===
SET PYTHON310_PATH="C:\Program Files\Python310\python.exe"

REM === Create virtual environment named tfenv ===
%PYTHON310_PATH% -m venv tfenv

REM === Activate the virtual environment ===
call tfenv\Scripts\activate

REM === Upgrade pip ===
python -m pip install --upgrade pip

REM === Install required packages ===
pip install tensorflow==2.13.0 opencv-python streamlit numpy scikit-learn matplotlib

echo.
echo ✅ Environment setup complete!
echo 🔁 To activate later, run: tfenv\Scripts\activate
pause
