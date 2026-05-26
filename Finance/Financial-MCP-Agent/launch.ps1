# Stock Investment Advisor - Launcher
$ErrorActionPreference = "SilentlyContinue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$FastAPIHost = "127.0.0.1"
$FastAPIPort = 8000
$StreamlitPort = 8501

Set-Location $ScriptDir

# --- Find Python ---
$PythonExe = $null
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
}
else {
    foreach ($cmd in @("python", "python3", "py")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) { $PythonExe = $found.Source; break }
    }
}

if (-not $PythonExe) {
    Write-Host "[ERROR] Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[INFO] Python: $PythonExe"

# --- Check dependencies ---
$depsOk = & $PythonExe -c "import uvicorn, streamlit, fastapi" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependencies missing. Run setup.bat first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Clean old processes on ports ---
$ports = @($FastAPIPort, $StreamlitPort)
foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        foreach ($c in $conn) {
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

# --- Start FastAPI ---
Write-Host ""
Write-Host "============================================================"
Write-Host "  Stock Investment Advisor"
Write-Host "============================================================"
Write-Host ""
Write-Host "[1/2] Starting FastAPI backend..."
$fastapiProc = Start-Process -FilePath $PythonExe `
    -ArgumentList "-u", "-m", "uvicorn", "src.api.app:app", "--host", $FastAPIHost, "--port", $FastAPIPort, "--log-level", "warning" `
    -WindowStyle Minimized `
    -PassThru
Write-Host "       FastAPI: http://${FastAPIHost}:${FastAPIPort}"

# Wait for FastAPI
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    try {
        $resp = Invoke-WebRequest -Uri "http://${FastAPIHost}:${FastAPIPort}/api/health" -TimeoutSec 2 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { break }
    } catch { }
}
Write-Host "       FastAPI ready"

# --- Start Streamlit ---
Write-Host "[2/2] Starting Streamlit frontend..."
$streamlitProc = Start-Process -FilePath $PythonExe `
    -ArgumentList "-u", "-m", "streamlit", "run", "src/app/Home.py", "--server.port", $StreamlitPort, "--server.headless", "true", "--browser.gatherUsageStats", "false" `
    -WindowStyle Minimized `
    -PassThru
Write-Host "       Streamlit: http://localhost:${StreamlitPort}"

# Wait for Streamlit
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:${StreamlitPort}" -TimeoutSec 2 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { break }
    } catch { }
}
Write-Host "       Streamlit ready"

# --- Open browser ---
Write-Host ""
Write-Host "============================================================"
Write-Host "  Opening browser..."
Start-Process "http://localhost:${StreamlitPort}"
Write-Host ""
Write-Host "  Web UI   : http://localhost:${StreamlitPort}"
Write-Host "  API Docs : http://${FastAPIHost}:${FastAPIPort}/docs"
Write-Host ""
Write-Host "  [Press Enter to stop all services]"
Write-Host "============================================================"
Write-Host ""

$null = Read-Host

# --- Stop services ---
Write-Host ""
Write-Host "Stopping services..."

# Kill the processes we started
if ($fastapiProc) {
    if (!$fastapiProc.HasExited) { $fastapiProc.Kill() }
    # Also kill child processes (worker processes spawned by uvicorn)
    & taskkill /f /t /pid $fastapiProc.Id 2>$null
}
if ($streamlitProc) {
    if (!$streamlitProc.HasExited) { $streamlitProc.Kill() }
    & taskkill /f /t /pid $streamlitProc.Id 2>$null
}

# Clean up by port (backup)
foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        foreach ($c in $conn) {
            & taskkill /f /t /pid $c.OwningProcess 2>$null
        }
    }
}

# Final sweep: kill any python process matching our service pattern
Get-WmiObject Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    $cmd = $_.CommandLine
    if ($cmd -match "uvicorn.*src\.api\.app" -or $cmd -match "streamlit.*Home\.py") {
        & taskkill /f /t /pid $_.ProcessId 2>$null
    }
}

Write-Host "All services stopped."
Start-Sleep -Seconds 2
