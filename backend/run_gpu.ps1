# GPU-only backend start (RTX 3050 / CUDA 12.x)
# Run: .\run_gpu.ps1

$ErrorActionPreference = "Stop"
$Backend = $PSScriptRoot
$Root = Split-Path -Parent $Backend
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$PythonArgs = @()
if (-not (Test-Path $Python)) {
    $Python = "py"
    $PythonArgs = @("-3.12")
}

$env:PYTHONPATH = $Backend
$env:PYTHONUNBUFFERED = "1"
$env:CUDA_MODULE_LOADING = "LAZY"

function Stop-PortListener([int]$Port) {
    $pids = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
    foreach ($procId in $pids) {
        if ($procId -and $procId -gt 0) {
            Write-Host "Stopping old server on port $Port (PID $procId)..." -ForegroundColor Yellow
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    if ($pids.Count -gt 0) { Start-Sleep -Seconds 1 }
}

Write-Host "Checking GPU..."
& $Python $PythonArgs -c @"
from core.gpu_runtime import init_gpu_runtime
d, half, name = init_gpu_runtime()
print(f'Ready: {name} | device={d} | fp16={half}')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "GPU check failed. Run setup_gpu.ps1 first." -ForegroundColor Red
    exit 1
}

Stop-PortListener -Port 5000

Set-Location $Backend
Write-Host ""
Write-Host "Starting backend on http://localhost:5000" -ForegroundColor Green
Write-Host "Keep this window open. Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# Flask writes its banner to stderr — do not treat that as a PowerShell error
$ErrorActionPreference = "Continue"
& $Python $PythonArgs -u predict_server.py
$exit = $LASTEXITCODE
if ($exit -ne 0) {
    Write-Host "Server exited with code $exit" -ForegroundColor Red
}
exit $exit
