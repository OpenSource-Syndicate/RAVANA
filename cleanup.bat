@echo off
setlocal enabledelayedexpansion

echo Deleting unwanted folders and files...

REM Delete folders
for /d /r %%d in (.venv __pycache__ .git) do (
    if exist "%%d" (
        echo Deleting folder: %%d
        rd /s /q "%%d"
    )
)

REM Delete files
for %%f in (uv.lock .gitignore .python-version) do (
    for /r %%x in (%%f) do (
        if exist "%%x" (
            echo Deleting file: %%x
            del /f /q "%%x"
        )
    )
)

endlocal
pause
