@echo off
echo ========================================
echo    AI TRADING SYSTEM - ONE CLICK START
echo ========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Start the complete trading system
python start_trading_system.py

pause