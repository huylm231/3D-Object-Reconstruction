"""
10_depth_estimation.py
Ước lượng depth (bản đồ chiều sâu) từ ảnh đơn bằng Depth-Anything-V2.

Lý thuyết (CVIP Chương 10):
- Monocular Depth Estimation: ước lượng depth từ 1 ảnh duy nhất bằng deep learning.
- Depth-Anything-V2: Vision Transformer foundation model, 4 kích cỡ (Small → Giant).
- Relative depth: xác định mối quan hệ gần/xa, không phải depth tuyệt đối.
- Ứng dụng 3D reconstruction: tạo depth map → chuyển sang point cloud → mesh.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys


def estimate_depth(
    image_path: str,
    encoder: str = "vits",
    input_size: int = 518,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ước lượng depth map bằng Depth-Anything-V2.

    Args:
        image_path: Đường dẫn ảnh đầu vào.
        encoder: Loại encoder ('vits' nhẹ nhất, 'vitb', 'vitl', 'vitg' nặng nhất).
        input_size: Kích thước input cho model (mặc định 518).

    Returns:
        (depth_float, depth_u8):
            - depth_float: depth map float [0, 1], shape (H, W).
            - depth_u8: depth map uint8 [0, 255], shape (H, W) — để hiển thị.
    """
    import torch
    from src.image_processing import PROJECT_ROOT

    # Thêm đường dẫn src vào sys.path để import depth_anything_v2
    src_dir = str(PROJECT_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from depth_anything_v2.dpt import DepthAnythingV2

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Config cho từng encoder
    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    model = DepthAnythingV2(**configs[encoder])

    # Tìm checkpoint
    checkpoint = PROJECT_ROOT / "models" / f"depth_anything_v2_{encoder}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint không tồn tại: {checkpoint}\n"
            f"Tải về tại: https://huggingface.co/depth-anything/Depth-Anything-V2-Small"
        )

    # [A5] weights_only=True để tránh thực thi mã độc qua pickle
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    # Inference
    raw_img = cv2.imread(str(image_path))
    if raw_img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    depth = model.infer_image(raw_img, input_size)

    # Áp dụng mặt nạ (mask) để xóa phông nền (gán depth = max - xa nhất)
    if mask is not None:
        if mask.shape != depth.shape:
            mask_resized = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        # [A6] Depth-Anything-V2 trả disparity (giá trị CAO = GẦN camera)
        # → gán nền = depth.min() (disparity thấp nhất = xa nhất)
        depth[mask_resized == 0] = depth.min()

    # Chuẩn hóa
    d_min, d_max = depth.min(), depth.max()
    depth_float = (depth - d_min) / (d_max - d_min + 1e-8)
    depth_u8 = (depth_float * 255).astype(np.uint8)

    return depth_float, depth_u8


def visualize_depth(depth_u8: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    Trực quan hóa depth map bằng colormap.

    Args:
        depth_u8: Depth map uint8 [0, 255].
        colormap: OpenCV colormap (INFERNO, JET, MAGMA, VIRIDIS, ...).

    Returns:
        Ảnh depth có màu (BGR).
    """
    return cv2.applyColorMap(depth_u8, colormap)


def save_depth(depth_u8: np.ndarray, path: str) -> None:
    """Lưu depth map ra file."""
    from src.image_processing.utils import save_image
    save_image(path, depth_u8)


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.10_depth_estimation
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang ước lượng depth cho ảnh: {img_path}")

    try:
        depth_float, depth_u8 = estimate_depth(img_path, encoder="vits")
        depth_color = visualize_depth(depth_u8)

        img = load_image(img_path)

        output_path = str(OUTPUT_DIR / "10_depth_estimation_result.png")
        show_images(
            [img, depth_u8, depth_color],
            ["Ảnh gốc", "Depth Map (Gray)", "Depth Map (Color)"],
            save_path=output_path,
        )

        save_depth(depth_u8, str(OUTPUT_DIR / "depth_map.png"))
        print(f"[DEMO] Kết quả đã lưu tại: {output_path}")

    except Exception as e:
        print(f"[DEMO] Lỗi: {e}")
        print("[DEMO] Đảm bảo đã có checkpoint tại models/depth_anything_v2_vits.pth")
