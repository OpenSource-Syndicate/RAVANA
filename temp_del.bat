@echo off
setlocal enabledelayedexpansion

echo Deleting __pycache__ and .pytest_cache directories...
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

echo Deleting knowledge files...
if exist "knowledge_id_map.pkl" (
    echo Deleting file: knowledge_id_map.pkl
    del /f /q "knowledge_id_map.pkl"
)
if exist "knowledge_index.faiss" (
    echo Deleting file: knowledge_index.faiss
    del /f /q "knowledge_index.faiss"
)

echo Deleting session directories...
if exist "chroma_db" (
    echo Deleting folder: chroma_db
    rd /s /q "chroma_db"
)
if exist "profiles" (
    echo Deleting folder: profiles
    rd /s /q "profiles"
)
if exist "shared_memory" (
    echo Deleting folder: shared_memory
    rd /s /q "shared_memory"
)

echo Cleanup complete.

endlocal
pause