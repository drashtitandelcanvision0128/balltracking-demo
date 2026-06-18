# GPU setup for Cricket Ball Trajectory (RTX 4050 / CUDA 12.x)
# Run from project root:  .\setup_gpu.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = "py -3.12"

Write-Host "Creating Python 3.12 venv at $venv ..."
& py -3.12 -m venv $venv

$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"

Write-Host "Installing PyTorch with CUDA 12.4 ..."
& $pip install --upgrade pip
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing app dependencies ..."
& $pip install ultralytics flask flask-cors opencv-python scipy werkzeug

Write-Host "Verifying GPU ..."
& $python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host ""
Write-Host "Done. Start GPU backend with:"
Write-Host "  cd backend"
Write-Host "  .\run_gpu.ps1"
