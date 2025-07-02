@echo off
title AGI System - 24/7 Continuous Operation
echo ================================================================================
echo                       AGI SYSTEM - 24/7 CONTINUOUS OPERATION
echo ================================================================================
echo.
echo Starting AGI System in continuous 24/7 mode...
echo.
echo This window will remain open while the AGI system is running.
echo Close this window or press Ctrl+C to stop the AGI system.
echo.
echo ================================================================================
echo.

:: Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Python not found in PATH. Checking for python3...
    where python3 >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Python not found. Please install Python and add it to your PATH.
        pause
        exit /b 1
    ) else (
        set PYTHON=python3
    )
) else (
    set PYTHON=python
)

:: Check if uv is available
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo UV not found, using standard Python.
    %PYTHON% run_agi_24_7.py --debug
) else (
    echo Using UV to run the AGI system.
    uv run run_agi_24_7.py --debug
)

:: If the script exits, pause to show any error messages
echo.
echo AGI System has stopped.
pause 