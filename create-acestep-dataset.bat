@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title "Upload ACE-Step Models Dataset to Kaggle"
cd /d "%~dp0"
py -3.10 -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); res = api.kernels_push(r'%~dp0kaggle\upload_acestep_dataset'); print('✓ Задача создания датасета успешно отправлена в Kaggle! Версия:', getattr(res, 'versionNumber', 'ok'))"
pause
