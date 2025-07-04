@echo off
setlocal enabledelayedexpansion

echo Deleting __pycache__..pytest_cache directories...
for /d /r %%d in (__pycache__,.pytest_cache) do (
    if exist "%%d" (
        echo Deleting folder: %%d
        rd /s /q "%%d"
    )
)

echo Deleting log files...
for /r %%f in (*.log) do (
    if exist "%%f" (
        echo Deleting file: %%f
        del /f /q "%%f"
    )
)
for /r %%f in (*.jsonl) do (
    if exist "%%f" (
        echo Deleting file: %%f
        del /f /q "%%f"
    )
)

echo Deleting database files...
for /r %%f in (*.db) do (
    if exist "%%f" (
        echo Deleting file: %%f
        del /f /q "%%f"
    )
)

echo Cleanup complete.

endlocal
pause
