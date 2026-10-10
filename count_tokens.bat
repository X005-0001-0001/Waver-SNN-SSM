@echo off
chcp 65001 >nul
cd /d "%~dp0"
python count_tokens.py
pause
