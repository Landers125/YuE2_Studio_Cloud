@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title YuE2 Studio - Kaggle
cd /d "%~dp0"

start "" "%~dp0YuE2-Studio-3.4.0-portable-windows-x64\YuE2-Studio.exe"

py -3.10 kaggle\headless_launcher.py %*
if errorlevel 1 pause
