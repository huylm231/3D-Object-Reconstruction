"""
01_color_spaces.py
Chuyển đổi không gian màu: RGB, Grayscale, HSV.
Cân bằng histogram (histogram equalization).

Lý thuyết (CVIP Chương 1):
- RGB: 3 kênh Red/Green/Blue, 8 bit/kênh → 24-bit color
- Grayscale: Gray = 0.2989*R + 0.5870*G + 0.1140*B
- HSV: Hue (0-360°), Saturation [0,1], Value [0,1]
- Ứng dụng 3D reconstruction: chuyển ảnh sang grayscale để tính SIFT;
  dùng HSV để segment theo màu (tách vật thể khỏi nền).
"""

import cv2
import numpy as np
from typing import Tuple


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh BGR sang ảnh xám (grayscale).
    Công thức: Gray = 0.2989*R + 0.5870*G + 0.1140*B
    """
    if len(img.shape) == 2:
        return img  # Đã là grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def convert_to_hsv(img: np.ndarray) -> np.ndarray:
    """
    Chuyển ảnh BGR sang không gian màu HSV.
    - H (Hue): 0-179 trong OpenCV (tương ứng 0-360°)
    - S (Saturation): 0-255
    - V (Value): 0-255
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def split_channels(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tách 3 kênh màu B, G, R từ ảnh BGR.

    Returns:
        Tuple (blue, green, red) — mỗi kênh là ảnh grayscale.
    """
    b, g, r = cv2.split(img)
    return b, g, r


def histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Cân bằng histogram — cải thiện độ tương phản ảnh.
    Dùng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    để tránh over-enhancement.

    Nếu ảnh màu: chuyển sang LAB, equalize kênh L, chuyển lại.
    Nếu ảnh xám: equalize trực tiếp.
    """
    if len(img.shape) == 2:
        # Ảnh grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    else:
        # Ảnh màu — xử lý trên kênh L (Lightness) trong không gian LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.01_color_spaces
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang xử lý ảnh: {img_path}")

    img = load_image(img_path)

    # Các phép biến đổi
    gray = convert_to_grayscale(img)
    hsv = convert_to_hsv(img)
    equalized = histogram_equalization(img)
    b, g, r = split_channels(img)

    # Hiển thị / lưu kết quả
    output_path = str(OUTPUT_DIR / "01_color_spaces_result.png")
    show_images(
        [img, gray, hsv, equalized],
        ["Ảnh gốc (BGR)", "Grayscale", "HSV", "CLAHE Equalized"],
        save_path=output_path,
    )
    print(f"[DEMO] Kết quả đã lưu tại: {output_path}")
