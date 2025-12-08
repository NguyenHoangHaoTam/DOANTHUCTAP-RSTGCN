@echo off
REM Script để chạy demo tự động với đúng virtual environment

REM Activate virtual environment (.venv)
call .venv\Scripts\activate.bat

REM Kiểm tra xem activate có thành công không
if errorlevel 1 (
    echo ERROR: Khong the activate virtual environment .venv
    echo Kiem tra xem thu muc .venv co ton tai khong
    pause
    exit /b 1
)

REM Chạy demo
python -m scripts.workflows.quick_demo

pause

