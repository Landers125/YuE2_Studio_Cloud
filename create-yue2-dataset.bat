@echo off
chcp 65001 >nul
title "Upload YuE2 Models Dataset to Kaggle"
cd /d "%~dp0"
py -3.10 -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); res = api.kernels_push(r'%~dp0kaggle\upload_yue2_dataset'); print('✓ Ядро создания датасета успешно отправлено в Kaggle! Версия:', getattr(res, 'versionNumber', 'ok'))"
if errorlevel 1 pause
