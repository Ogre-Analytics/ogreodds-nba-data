@echo off
title NBA Predictions Export
cd /d c:\Users\nicke\nba-betting-tool
python predict.py
echo.
echo ============================================================
echo  Done. predictions.json has been updated.
echo  Press any key to close.
echo ============================================================
pause >nul
