@echo off
REM ============================================================
REM setup.bat — Windows 环境初始化脚本
REM ============================================================

setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PARENT_DIR=%PROJECT_DIR%\..\.."
set "VENV_DIR=%PARENT_DIR%\venv"

echo ============================================================
echo   Stock Investment Advisor - Windows 环境初始化
echo ============================================================
echo.

REM ── 查找 Python ──────────────────
set "PYTHON_EXE="
for %%c in (python python3 py) do (
    if not defined PYTHON_EXE (
        where %%c >nul 2>&1 && set "PYTHON_EXE=%%c"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] 未找到 Python，请先安装 Python 3.10+
    echo         下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('%PYTHON_EXE% --version 2^>^&1') do set "PY_VER=%%v"
echo [INFO]  使用 Python: %PYTHON_EXE% (版本 %PY_VER%)
echo.

REM ── 检查是否为 WSL venv ──────────
if exist "%VENV_DIR%\bin\python" (
    if not exist "%VENV_DIR%\Scripts\python.exe" (
        echo [WARN]  检测到 WSL/Linux 虚拟环境，无法在 Windows 上使用
        echo [WARN]  正在删除旧 venv...
        rmdir /s /q "%VENV_DIR%" 2>nul
        echo [INFO]  旧 venv 已删除
        echo.
    )
)

REM ── 创建虚拟环境 ────────────────
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO]  创建 Windows 虚拟环境...
    %PYTHON_EXE% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo [INFO]  虚拟环境已创建
) else (
    echo [INFO]  虚拟环境已存在
)
echo.

REM ── 激活虚拟环境并安装依赖 ──────
echo [INFO]  激活虚拟环境并安装依赖...
call "%VENV_DIR%\Scripts\activate.bat"

echo [INFO]  升级 pip...
python -m pip install --upgrade pip -q

echo [INFO]  安装项目依赖...
pip install -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] 依赖安装失败
    pause
    exit /b 1
)
echo.

echo ============================================================
echo   初始化完成！
echo.
echo   启动服务:   cd /d "%PROJECT_DIR%" ^&^& run.bat
echo   或使用 PowerShell: cd "%PROJECT_DIR%"; .\run.ps1
echo.
echo   访问 Web UI: http://localhost:8501
echo   API 文档:    http://127.0.0.1:8000/docs
echo ============================================================
echo.
if /i not "%~1"=="--silent" (
    pause
)
exit /b 0
