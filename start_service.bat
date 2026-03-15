@echo off
echo ========================================
echo Vehicle Counter Service
echo ========================================
echo.

cd /d %~dp0

echo Activating virtual environment...
call ..\venv\Scripts\activate.bat 2>nul || call venv\Scripts\activate.bat 2>nul

echo.
echo Starting service...
echo Press Ctrl+C to stop
echo.

python vehicle_counter_service.py

pause
