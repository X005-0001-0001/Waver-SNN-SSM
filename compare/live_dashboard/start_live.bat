@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Live Dashboard  (port 8778)
echo ============================================
echo.
echo   http://127.0.0.1:8778/
echo.
echo   Close this window to stop.
echo.
python server.py 8778 1
pause
