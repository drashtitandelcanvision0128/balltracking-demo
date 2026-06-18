# Start production FastAPI server (port 8000) — GPU required
$Backend = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $Backend
$env:PYTHONPATH = $Backend
$env:CUDA_MODULE_LOADING = "LAZY"
Set-Location $Backend
& "$Root\.venv\Scripts\python.exe" -c "from core.gpu_runtime import init_gpu_runtime; init_gpu_runtime()"
if ($LASTEXITCODE -ne 0) { exit 1 }
& "$Root\.venv\Scripts\python.exe" -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
