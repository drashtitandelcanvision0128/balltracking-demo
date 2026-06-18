# GPU-only backend start (RTX 4050 / CUDA 12.x)
# Run: .\run_gpu.ps1

$ErrorActionPreference = "Stop"
$Backend = $PSScriptRoot
$Root = Split-Path -Parent $Backend
$Python = Join-Path $Root ".venv\Scripts\python.exe"

$env:PYTHONPATH = $Backend
$env:CUDA_MODULE_LOADING = "LAZY"

Write-Host "Checking GPU..."
& $Python -c @"
from core.gpu_runtime import init_gpu_runtime
d, half, name = init_gpu_runtime()
print(f'Ready: {name} | device={d} | fp16={half}')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "GPU check failed. Run setup_gpu.ps1 first." -ForegroundColor Red
    exit 1
}

Set-Location $Backend
Write-Host "Starting GPU server on http://localhost:5000 ..."
& $Python predict_server.py
