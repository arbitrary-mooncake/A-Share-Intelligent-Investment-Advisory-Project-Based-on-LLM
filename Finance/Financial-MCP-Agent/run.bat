@echo off
REM ============================================================
REM run.bat — Windows CMD 一键启动 FastAPI + Streamlit
REM
REM 用法:
REM   run.bat          REM 启动两个服务
REM   run.bat stop     REM 停止两个服务
REM   run.bat status   REM 查看服务状态
REM ============================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "FASTAPI_HOST=127.0.0.1"
set "FASTAPI_PORT=8000"
set "STREAMLIT_PORT=8501"

set "PID_DIR=%PROJECT_DIR%\.run"
set "FASTAPI_PID=%PID_DIR%\fastapi.pid"
set "STREAMLIT_PID=%PID_DIR%\streamlit.pid"
set "LOG_DIR=%PID_DIR%\logs"

if not exist "%PID_DIR%" mkdir "%PID_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM ── 查找 Python ──────────────────
set "PYTHON_EXE="
for %%c in (python python3 py) do (
    if not defined PYTHON_EXE (
        where %%c >nul 2>&1 && set "PYTHON_EXE=%%c"
    )
)

REM 优先使用虚拟环境（Windows venv）
set "VENV_WIN_PYTHON=%PROJECT_DIR%\..\..\venv\Scripts\python.exe"
set "VENV_WSL_PYTHON=%PROJECT_DIR%\..\..\venv\bin\python"

if exist "%VENV_WIN_PYTHON%" (
    set "PYTHON_EXE=%VENV_WIN_PYTHON%"
) else if exist "%VENV_WSL_PYTHON%" (
    echo [WARN]  检测到 WSL/Linux 虚拟环境，无法在 Windows 上运行
    echo [WARN]  请重新创建 Windows 虚拟环境:
    echo [WARN]    1. 删除旧 venv: rmdir /s /q "..\..\venv"
    echo [WARN]    2. 创建新 venv: python -m venv "..\..\venv"
    echo [WARN]    3. 激活 venv:   "..\..\venv\Scripts\activate.bat"
    echo [WARN]    4. 安装依赖:   pip install -r requirements.txt
    echo [INFO]  回退使用系统 Python
)

if not defined PYTHON_EXE (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    exit /b 1
)

echo [INFO]  使用 Python: %PYTHON_EXE%

REM ── 检查进程是否运行 ─────────────
if "%1"=="" goto :start
if "%1"=="start" goto :start
if "%1"=="stop" goto :stop
if "%1"=="status" goto :status
if "%1"=="restart" goto :restart
echo 用法: %0 {start^|stop^|restart^|status}
exit /b 1

REM ── 启动 ─────────────────────────
:start
echo [INFO]  启动服务...

REM 检查 FastAPI
call :check_pid "%FASTAPI_PID%" "FastAPI"
if !running!==1 (
    echo [WARN]  FastAPI 已在运行 ^(PID !pid!^)
) else (
    echo [INFO]  启动 FastAPI 后端 -^> http://%FASTAPI_HOST%:%FASTAPI_PORT%
    start /B "" "%PYTHON_EXE%" -u -m uvicorn src.api.app:app --host %FASTAPI_HOST% --port %FASTAPI_PORT% --log-level info > "%LOG_DIR%\fastapi.log" 2>&1
    set "fp=!errorlevel!"
    REM 获取 uvicorn 进程 PID（start /B 不直接返回 PID，用 tasklist 查找）
    for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo table /nh 2^>nul ^| findstr /r "[0-9]"') do (
        set "last_pid=%%a"
    )
    echo !last_pid! > "%FASTAPI_PID%"
    echo [INFO]  FastAPI PID: !last_pid!
)

REM 检查 Streamlit
call :check_pid "%STREAMLIT_PID%" "Streamlit"
if !running!==1 (
    echo [WARN]  Streamlit 已在运行 ^(PID !pid!^)
) else (
    echo [INFO]  启动 Streamlit 前端 -^> http://localhost:%STREAMLIT_PORT%
    start /B "" "%PYTHON_EXE%" -u -m streamlit run "%PROJECT_DIR%\src\app\Home.py" --server.port %STREAMLIT_PORT% --server.headless true --browser.gatherUsageStats false > "%LOG_DIR%\streamlit.log" 2>&1
    for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo table /nh 2^>nul ^| findstr /r "[0-9]"') do (
        set "last_pid=%%a"
    )
    echo !last_pid! > "%STREAMLIT_PID%"
    echo [INFO]  Streamlit PID: !last_pid!
)

echo.
echo [INFO]  等待服务启动...
timeout /t 3 /nobreak >nul
echo [INFO]  访问 Web UI: http://localhost:%STREAMLIT_PORT%
echo [INFO]  API 文档:    http://%FASTAPI_HOST%:%FASTAPI_PORT%/docs
goto :eof

REM ── 停止 ─────────────────────────
:stop
echo [INFO]  停止服务...
call :kill_pid "%FASTAPI_PID%" "FastAPI"
call :kill_pid "%STREAMLIT_PID%" "Streamlit"
echo [INFO]  服务已停止
goto :eof

REM ── 重启 ─────────────────────────
:restart
call :stop
timeout /t 2 /nobreak >nul
call :start
goto :eof

REM ── 状态 ─────────────────────────
:status
set "any=0"
call :check_pid "%FASTAPI_PID%" "FastAPI"
if !running!==1 (
    echo [INFO]  FastAPI 运行中  ^(PID !pid!^) -^> http://%FASTAPI_HOST%:%FASTAPI_PORT%/docs
    set "any=1"
) else (
    echo [ERROR] FastAPI 未运行
)
call :check_pid "%STREAMLIT_PID%" "Streamlit"
if !running!==1 (
    echo [INFO]  Streamlit 运行中 ^(PID !pid!^) -^> http://localhost:%STREAMLIT_PORT%
    set "any=1"
) else (
    echo [ERROR] Streamlit 未运行
)
if !any!==0 (
    echo.
    echo [INFO]  使用 'run.bat' 启动服务
)
goto :eof

REM ── 子函数: 检查 PID 文件 ────────
:check_pid
set "running=0"
set "pid="
if not exist "%~1" goto :eof
set /p "pid=" < "%~1"
tasklist /fi "PID eq !pid!" 2>nul | find "!pid!" >nul 2>&1
if !errorlevel!==0 set "running=1"
goto :eof

REM ── 子函数: 终止进程 ────────────
:kill_pid
set "pid="
if not exist "%~1" goto :eof
set /p "pid=" < "%~1"
echo [INFO]  停止 %~2 ^(PID !pid!^)...
taskkill /PID !pid! /F >nul 2>&1
del "%~1" >nul 2>&1
goto :eof
