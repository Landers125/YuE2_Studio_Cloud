@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title "Stop Kaggle Instance"
cd /d "%~dp0"
py -3.10 kaggle\stop_server.py
pause
