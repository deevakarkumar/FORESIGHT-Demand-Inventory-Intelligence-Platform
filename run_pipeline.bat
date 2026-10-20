@echo off
title FORESIGHT Full Analytics Pipeline
echo ====================================================
echo Running FORESIGHT Data Pipeline & Prophet Models...
echo ====================================================
cd /d "%~dp0"
python main.py
pause
