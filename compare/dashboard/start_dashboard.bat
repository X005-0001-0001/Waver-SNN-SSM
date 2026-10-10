@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Compare Dashboard
echo ============================================
echo.
echo   http://127.0.0.1:8777/
echo.
echo   Close this window to stop.
echo.
python server.py 8777 1
pause
