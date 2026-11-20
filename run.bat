@echo off
title Smart Waste Classification System
echo ========================================================
echo Starting Smart Waste Classification Web Server...
echo ========================================================

cd /d "%~dp0"

IF EXIST "..\.venv\Scripts\python.exe" (
    SET "PY_PATH=..\.venv\Scripts\python.exe"
) ELSE IF EXIST ".venv\Scripts\python.exe" (
    SET "PY_PATH=.venv\Scripts\python.exe"
) ELSE (
    SET "PY_PATH=python"
)

echo Using Python from: %PY_PATH%
"%PY_PATH%" app.py

pause
