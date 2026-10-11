@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title "YuE2 & ACE-Step Studio - Kaggle 2x T4"
cd /d "%~dp0"
py -3.10 kaggle\headless_launcher.py --app both %*
if errorlevel 1 pause
