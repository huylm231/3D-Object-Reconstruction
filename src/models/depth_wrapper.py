# models/depth_wrapper.py
"""
Wrapper don gian cho Depth-Anything-V2.
Su dung khi da cai dat du thu vien (torch, opencv-python).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent # Project root
DEPTH_SRC = ROOT / "src" / "depth_anything_v2"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_depth_model(encoder: str = "vits"):
    """
    Tai mo hinh Depth-Anything-V2.
    encoder: 'vits' (nhe), 'vitb', 'vitl', 'vitg' (nang)
    """
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from src.depth_anything_v2.dpt import DepthAnythingV2

    configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = DepthAnythingV2(**configs[encoder])
    ckpt_candidates = [
        ROOT / "models" / f"depth_anything_v2_{encoder}.pth",
        ROOT / "weights" / f"depth_anything_v2_{encoder}.pth",
    ]
    ckpt = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt is None:
        raise FileNotFoundError(
            "Checkpoint Depth-Anything-V2 không tìm thấy. Kiểm tra thư mục models/ hoặc weights/.\n"
            f"Candidates:\n  - {ckpt_candidates[0]}\n  - {ckpt_candidates[1]}\n"
            "Download from: https://huggingface.co/depth-anything/Depth-Anything-V2-Small"
        )

    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    return model.to(device).eval(), device


def infer_depth(model, device, image_bgr, input_size: int = 518):
    """
    Chay inference tren 1 anh BGR (numpy array).
    Tra ve depth map float [0, 1], shape (H, W).
    """
    import numpy as np
    depth = model.infer_image(image_bgr, input_size)
    d_min, d_max = depth.min(), depth.max()
    return (depth - d_min) / (d_max - d_min + 1e-8)
