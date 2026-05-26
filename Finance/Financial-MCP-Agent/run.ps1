# ============================================================
# run.ps1 — Windows PowerShell 一键启动 FastAPI + Streamlit
#
# 用法:
#   .\run.ps1          # 启动两个服务
#   .\run.ps1 stop     # 停止两个服务
#   .\run.ps1 status   # 查看服务状态
#   .\run.ps1 restart  # 重启两个服务
# ============================================================

param(
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"

# ── 路径配置 ──────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = $ScriptDir
$FastAPIEntry = "$ProjectDir\src\api\app.py"
$StreamlitEntry = "$ProjectDir\src\app\Home.py"
$FastAPIHost = "127.0.0.1"
$FastAPIPort = 8000
$StreamlitPort = 8501

# PID 文件目录
$PIDDir = "$ProjectDir\.run"
$FastAPIPIDFile = "$PIDDir\fastapi.pid"
$StreamlitPIDFile = "$PIDDir\streamlit.pid"

# 日志目录
$LogDir = "$PIDDir\logs"
New-Item -ItemType Directory -Force -Path $PIDDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ── 辅助函数 ──────────────────────────────────────────────
function Write-Info {
    Write-Host "[INFO]  $args" -ForegroundColor Green
}

function Write-Warn {
    Write-Host "[WARN]  $args" -ForegroundColor Yellow
}

function Write-ErrorMsg {
    Write-Host "[ERROR] $args" -ForegroundColor Red
}

function Test-ProcessRunning {
    param([string]$PIDFile)
    if (-not (Test-Path $PIDFile)) { return $false }
    try {
        $pid = Get-Content $PIDFile -Raw
        $proc = Get-Process -Id $pid -ErrorAction Stop
        return -not $proc.HasExited
    } catch {
        return $false
    }
}

function Stop-ProcessByPIDFile {
    param([string]$PIDFile, [string]$Name)
    if (-not (Test-Path $PIDFile)) {
        Write-Warn "$Name PID 文件不存在"
        return $false
    }
    try {
        $pid = Get-Content $PIDFile -Raw
        $proc = Get-Process -Id $pid -ErrorAction Stop
        Write-Info "停止 $Name (PID $pid)..."
        $proc.Kill()
        $proc.WaitForExit(5000)
        Remove-Item $PIDFile -Force
        return $true
    } catch {
        Remove-Item $PIDFile -Force -ErrorAction SilentlyContinue
        return $false
    }
}

# ── 检查 Python 依赖 ─────────────────────────────────────
function Check-Deps {
    $pythonCmd = $null
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $result = & $cmd --version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $pythonCmd = $cmd
                break
            }
        } catch { }
    }
    if (-not $pythonCmd) {
        Write-ErrorMsg "未找到 Python，请先安装 Python 3.10+"
        exit 1
    }
    Write-Info "使用 Python: $pythonCmd ($(& $pythonCmd --version 2>&1))"

    # 检查是否有 Windows 虚拟环境，若只有 WSL venv 则给出指引
    $venvWinPython = "$ProjectDir\..\..\venv\Scripts\python.exe"
    $venvWslPython = "$ProjectDir\..\..\venv\bin\python"
    if (Test-Path $venvWinPython) {
        $script:PythonExe = $venvWinPython
        Write-Info "使用虚拟环境 Python: $venvWinPython"
    } elseif (Test-Path $venvWslPython) {
        Write-Warn "检测到 WSL/Linux 虚拟环境 (venv/bin/python)，无法在 Windows 上运行"
        Write-Warn "请重新创建 Windows 虚拟环境："
        Write-Warn "  1. 删除旧 venv: Remove-Item -Recurse -Force ..\..\venv"
        Write-Warn "  2. 创建新 venv: python -m venv ..\..\venv"
        Write-Warn "  3. 激活 venv:   ..\..\venv\Scripts\Activate.ps1"
        Write-Warn "  4. 安装依赖:   pip install -r requirements.txt"
        Write-Info "回退使用系统 Python: $pythonCmd"
        $script:PythonExe = $pythonCmd
    } else {
        $script:PythonExe = $pythonCmd
    }

    # 检查关键包
    $packages = & $script:PythonExe -c "import uvicorn, streamlit; print('ok')" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-ErrorMsg "缺少依赖 (uvicorn/streamlit)"
        Write-Info "请激活虚拟环境后运行: pip install -r requirements.txt"
        exit 1
    }
    Write-Info "依赖检查通过"
}

# ── 启动服务 ─────────────────────────────────────────────
function Start-Services {
    Check-Deps

    $started = $false

    # 启动 FastAPI
    if (Test-ProcessRunning $FastAPIPIDFile) {
        Write-Warn "FastAPI 已在运行 (PID $(Get-Content $FastAPIPIDFile -Raw))"
    } else {
        Write-Info "启动 FastAPI 后端 → http://${FastAPIHost}:${FastAPIPort}"
        $fastapiLog = "$LogDir\fastapi.log"
        $proc = Start-Process -FilePath $script:PythonExe `
            -ArgumentList "-u", "-m", "uvicorn", "src.api.app:app", "--host", $FastAPIHost, "--port", $FastAPIPort, "--log-level", "info" `
            -WorkingDirectory $ProjectDir `
            -NoNewWindow `
            -RedirectStandardOutput $fastapiLog `
            -RedirectStandardError "$LogDir\fastapi_err.log" `
            -PassThru
        $proc.Id | Out-File -FilePath $FastAPIPIDFile -NoNewline
        Write-Info "FastAPI PID: $($proc.Id)"
        $started = $true
    }

    # 启动 Streamlit
    if (Test-ProcessRunning $StreamlitPIDFile) {
        Write-Warn "Streamlit 已在运行 (PID $(Get-Content $StreamlitPIDFile -Raw))"
    } else {
        Write-Info "启动 Streamlit 前端 → http://localhost:${StreamlitPort}"
        $streamlitLog = "$LogDir\streamlit.log"
        $proc = Start-Process -FilePath $script:PythonExe `
            -ArgumentList "-u", "-m", "streamlit", "run", $StreamlitEntry, "--server.port", $StreamlitPort, "--server.headless", "true", "--browser.gatherUsageStats", "false" `
            -WorkingDirectory $ProjectDir `
            -NoNewWindow `
            -RedirectStandardOutput $streamlitLog `
            -RedirectStandardError "$LogDir\streamlit_err.log" `
            -PassThru
        $proc.Id | Out-File -FilePath $StreamlitPIDFile -NoNewline
        Write-Info "Streamlit PID: $($proc.Id)"
        $started = $true
    }

    if ($started) {
        Write-Info "等待服务启动..."
        Start-Sleep -Seconds 3

        # 检查启动结果
        if (Test-ProcessRunning $FastAPIPIDFile) {
            Write-Info "FastAPI 运行中  → http://${FastAPIHost}:${FastAPIPort}/docs"
        } else {
            Write-ErrorMsg "FastAPI 启动失败，请查看日志: $LogDir\fastapi_err.log"
        }

        if (Test-ProcessRunning $StreamlitPIDFile) {
            Write-Info "Streamlit 运行中 → http://localhost:${StreamlitPort}"
        } else {
            Write-ErrorMsg "Streamlit 启动失败，请查看日志: $LogDir\streamlit_err.log"
        }
    }

    Write-Host ""
    Write-Info "访问 Web UI: http://localhost:${StreamlitPort}"
    Write-Info "API 文档:    http://${FastAPIHost}:${FastAPIPort}/docs"
}

# ── 停止服务 ─────────────────────────────────────────────
function Stop-Services {
    $stopped = $false

    if (Stop-ProcessByPIDFile $FastAPIPIDFile "FastAPI") {
        $stopped = $true
    }
    if (Stop-ProcessByPIDFile $StreamlitPIDFile "Streamlit") {
        $stopped = $true
    }

    # 清理残留进程
    Get-Process -Name "python" -ErrorAction SilentlyContinue | ForEach-Object {
        $cmd = (Get-WmiObject Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine
        if ($cmd -match "uvicorn.*src\.api\.app" -or $cmd -match "streamlit.*Home\.py") {
            Write-Info "清理残留进程: $($_.Id)"
            Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            $stopped = $true
        }
    }

    if ($stopped) {
        Write-Info "服务已停止"
    } else {
        Write-Warn "未发现运行中的服务"
    }
}

# ── 状态检查 ─────────────────────────────────────────────
function Show-Status {
    $anyRunning = $false

    if (Test-ProcessRunning $FastAPIPIDFile) {
        $pid = (Get-Content $FastAPIPIDFile -Raw).Trim()
        Write-Info "FastAPI 运行中  (PID $pid) → http://${FastAPIHost}:${FastAPIPort}/docs"
        $anyRunning = $true
    } else {
        Write-ErrorMsg "FastAPI 未运行"
    }

    if (Test-ProcessRunning $StreamlitPIDFile) {
        $pid = (Get-Content $StreamlitPIDFile -Raw).Trim()
        Write-Info "Streamlit 运行中 (PID $pid) → http://localhost:${StreamlitPort}"
        $anyRunning = $true
    } else {
        Write-ErrorMsg "Streamlit 未运行"
    }

    if (-not $anyRunning) {
        Write-Host ""
        Write-Info "使用 '.\run.ps1 start' 启动服务"
    }
}

# ── 主入口 ───────────────────────────────────────────────
switch ($Action) {
    "start"   { Start-Services }
    "stop"    { Stop-Services }
    "restart" { Stop-Services; Start-Sleep -Seconds 2; Start-Services }
    "status"  { Show-Status }
    default   {
        Write-Host "用法: .\run.ps1 {start|stop|restart|status}"
        exit 1
    }
}
