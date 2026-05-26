@echo off
chcp 65001 >nul 2>&1
title Stock Investment Advisor
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "STREAMLIT_PORT=8501"

cd /d "%SCRIPT_DIR%"

rem --- Find Python (prefer venv) ---
set "PYTHON_EXE="
set "VENV_PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" set "PYTHON_EXE=%VENV_PYTHON%"
if not defined PYTHON_EXE (
    for %%c in (python python3 py) do (
        if not defined PYTHON_EXE (
            where %%c >nul 2>&1 && set "PYTHON_EXE=%%c"
        )
    )
)
if not defined PYTHON_EXE (
    echo [ERROR] Python not found. Install Python 3.10+
    pause
    exit /b 1
)
echo [INFO] Python: %PYTHON_EXE%

rem --- Check dependencies ---
%PYTHON_EXE% -c "import uvicorn, streamlit, fastapi" 2>nul
if errorlevel 1 (
    echo [ERROR] Dependencies missing. Run setup.bat first.
    pause
    exit /b 1
)

rem --- Clean old processes ---
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000.*LISTENING"') do taskkill /pid %%a /f >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8501.*LISTENING"') do taskkill /pid %%a /f >nul 2>&1

rem --- Start FastAPI ---
echo.
echo ============================================================
echo   Stock Investment Advisor
echo ============================================================
echo.
echo [1/2] Starting FastAPI backend...
start "Stock Advisor - FastAPI" /MIN "%PYTHON_EXE%" -u -m uvicorn src.api.app:app --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level warning
echo        FastAPI: http://%FASTAPI_HOST%:%FASTAPI_PORT%

rem --- Wait for FastAPI ---
set "N=0"
:wait_api
timeout /t 1 /nobreak >nul
set /a N+=1
%PYTHON_EXE% -c "import urllib.request; urllib.request.urlopen('http://%FASTAPI_HOST%:%FASTAPI_PORT%/api/health', timeout=2)" 2>nul && goto api_ready
if %N% lss 20 goto wait_api
echo [WARN] FastAPI startup slow, continuing...
:api_ready
echo        FastAPI ready

rem --- Start Streamlit ---
echo [2/2] Starting Streamlit frontend...
start "Stock Advisor - Streamlit" /MIN "%PYTHON_EXE%" -u -m streamlit run src/app/Home.py --server.port %STREAMLIT_PORT% --server.headless true --browser.gatherUsageStats false
echo        Streamlit: http://localhost:%STREAMLIT_PORT%

rem --- Wait for Streamlit ---
set "N=0"
:wait_st
timeout /t 2 /nobreak >nul
set /a N+=2
%PYTHON_EXE% -c "import urllib.request; urllib.request.urlopen('http://localhost:%STREAMLIT_PORT%', timeout=2)" 2>nul && goto st_ready
if %N% lss 60 goto wait_st
echo [WARN] Streamlit startup slow, continuing...
:st_ready
echo        Streamlit ready

rem --- Open browser ---
echo.
echo ============================================================
echo   Opening browser...
start "" http://localhost:%STREAMLIT_PORT%
echo.
echo   Web UI   : http://localhost:%STREAMLIT_PORT%
echo   API Docs : http://%FASTAPI_HOST%:%FASTAPI_PORT%/docs
echo.
echo   [Press any key to stop all services]
echo ============================================================
echo.

pause >nul

rem --- Stop services ---
echo.
echo Stopping services...
taskkill /fi "WINDOWTITLE eq Stock Advisor - FastAPI*" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq Stock Advisor - Streamlit*" /f >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8000.*LISTENING"') do taskkill /pid %%a /f >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8501.*LISTENING"') do taskkill /pid %%a /f >nul 2>&1
echo All services stopped.
timeout /t 2 /nobreak >nul
