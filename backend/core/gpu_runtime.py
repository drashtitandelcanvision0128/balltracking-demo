"""
GPU runtime — force CUDA, tune cuDNN, and expose fast-inference settings.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from core.config import CONFIG

_GPU_CFG = CONFIG.get("gpu", {})
_INITIALIZED = False
_DEVICE = "cpu"
_HALF = False
_GPU_NAME: str | None = None


def require_gpu() -> None:
    """Exit if CUDA is not available (user wants GPU-only runs)."""
    try:
        import torch
    except ImportError as exc:
        print("[GPU] PyTorch not installed.", file=sys.stderr)
        raise SystemExit(1) from exc

    if not torch.cuda.is_available():
        print(
            "[GPU] CUDA not available. Install CUDA PyTorch:\n"
            "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
            file=sys.stderr,
        )
        raise SystemExit(1)


def init_gpu_runtime() -> tuple[int | str, bool, str | None]:
    """
    One-time GPU setup: cuDNN benchmark, TF32, memory fraction.
    Returns (device, half_precision, gpu_name).
    """
    global _INITIALIZED, _DEVICE, _HALF, _GPU_NAME
    if _INITIALIZED:
        return _DEVICE, _HALF, _GPU_NAME

    require_gpu()

    import torch

    if _GPU_CFG.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if _GPU_CFG.get("cudnn_benchmark", True):
        torch.backends.cudnn.benchmark = True

    device_id = int(_GPU_CFG.get("device_id", 0))
    _DEVICE = device_id
    _HALF = bool(_GPU_CFG.get("half_precision", True))
    _GPU_NAME = torch.cuda.get_device_name(device_id)

    # Reserve headroom for display / other apps on laptop GPUs
    fraction = float(_GPU_CFG.get("memory_fraction", 0.85))
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device_id)
    except Exception:
        pass

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    _INITIALIZED = True
    print(f"[GPU] {_GPU_NAME} | half={_HALF} | cudnn_benchmark={_GPU_CFG.get('cudnn_benchmark', True)}")
    return _DEVICE, _HALF, _GPU_NAME


def load_yolo_model(model_path: str):
    """Load YOLO on GPU with fuse + warmup at configured image sizes."""
    from ultralytics import YOLO

    device, half, gpu_name = init_gpu_runtime()
    model = YOLO(model_path)
    model.to(device)

    if _GPU_CFG.get("fuse_layers", True):
        try:
            model.fuse()
            print("[GPU] Model layers fused")
        except Exception as exc:
            print(f"[GPU] Fuse skipped: {exc}")

    warmup_sz = int(_GPU_CFG.get("warmup_imgsz", 640))
    dummy = np.zeros((warmup_sz, warmup_sz, 3), dtype=np.uint8)
    model.predict(
        dummy,
        verbose=False,
        device=device,
        half=half,
        imgsz=warmup_sz,
        augment=False,
    )
    print(f"[GPU] Warmup done imgsz={warmup_sz}")
    return model, half, device, gpu_name


def infer_settings(ball_active: bool) -> dict:
    """Return imgsz / max_dim for current pipeline state."""
    if ball_active:
        return {
            "imgsz": int(_GPU_CFG.get("active_imgsz", 960)),
            "max_dim": int(_GPU_CFG.get("active_max_dim", 960)),
            "stride": 1,
        }
    return {
        "imgsz": int(_GPU_CFG.get("waiting_imgsz", 640)),
        "max_dim": int(_GPU_CFG.get("waiting_max_dim", 640)),
        "stride": int(_GPU_CFG.get("waiting_stride", 3)),
    }


def ffmpeg_encode_args(input_path: str, output_path: str) -> list[str]:
    """Prefer NVENC on NVIDIA GPUs for fast post-encode."""
    ffmpeg = CONFIG["processing"].get("ffmpeg_path", "ffmpeg")
    use_nvenc = bool(_GPU_CFG.get("nvenc", True)) and _GPU_NAME is not None
    if use_nvenc:
        return [
            ffmpeg, "-y", "-hwaccel", "cuda", "-i", input_path,
            "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path,
        ]
    return [
        ffmpeg, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path,
    ]
