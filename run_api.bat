@echo off
title FORESIGHT FastAPI Scoring Service
echo ====================================================
echo Starting FORESIGHT FastAPI Scoring Service...
echo Swagger UI will be available at http://localhost:8000/docs
echo ====================================================
cd /d "%~dp0"
python -m uvicorn service.api:app --host 0.0.0.0 --port 8000
pause
