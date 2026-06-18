# Fast GPU training (RTX 4050) - ~1-3 hours for 100 epochs vs ~2 days on CPU
# Run from backend: .\scripts\train_fast_gpu.ps1

$ErrorActionPreference = "Stop"
$Backend = Split-Path -Parent $PSScriptRoot
$Root = Split-Path -Parent $Backend
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "GPU venv missing. Run from project root: .\setup_gpu.ps1" -ForegroundColor Red
    exit 1
}

Set-Location $Backend
$env:PYTHONPATH = $Backend
$env:CUDA_MODULE_LOADING = "LAZY"

Write-Host "Verifying GPU..."
& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

Write-Host "Starting FAST GPU training (100 epochs, yolov8m, imgsz=960)..."
& $Python scripts/train_improved.py `
    --data data.yaml `
    --epochs 100 `
    --model yolov8s.pt `
    --imgsz 960 `
    --batch 6 `
    --device 0 `
    --name train_improved `
    --cache

Write-Host "Done. Restart backend: .\run_gpu.ps1"
