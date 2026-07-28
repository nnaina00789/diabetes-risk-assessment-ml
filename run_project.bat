@echo off
title Diabetes Risk Assessment System
color 0A
cls

echo =========================================================================
echo               DIABETES RISK ASSESSMENT MACHINE LEARNING SYSTEM          
echo =========================================================================
echo.
echo Launching Python Application Server...
echo Opening http://localhost:5000 in your browser...
echo.

cd /d "%~dp0"

python server.py
if %ERRORLEVEL% NEQ 0 (
    echo Python command failed, trying fallback 'py'...
    py server.py
)

pause
