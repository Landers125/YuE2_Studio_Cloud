@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3.10 colab\local_bridge.py --app yue2 %*
if errorlevel 1 pause
