@echo off
title FORESIGHT Dashboard Launcher
echo ====================================================
echo Starting FORESIGHT Demand & Inventory Intelligence Dashboard...
echo ====================================================
cd /d "%~dp0"
python -m streamlit run app.py
pause
