"""
05_morphology.py
Xử lý hình thái học: Erosion, Dilation, Opening, Closing, Gradient.

Lý thuyết (CVIP Chương 2):
- Erosion (xói mòn): thu nhỏ đối tượng, loại nhiễu nhỏ.
- Dilation (giãn nở): phóng to đối tượng, lấp lỗ nhỏ.
- Opening = erosion → dilation: loại nhiễu nhỏ, làm mịn viền ngoài.
- Closing = dilation → erosion: lấp lỗ hổng nhỏ, làm mịn viền trong.
- Gradient hình thái: tìm biên đối tượng.
- Ứng dụng 3D reconstruction: tạo silhouette/mask sạch trước khi dựng Visual Hull.
"""

import cv2
import numpy as np
from typing import Tuple


def erode(img: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Xói mòn (Erosion) — thu nhỏ vùng trắng, loại bỏ nhiễu nhỏ.

    Args:
        img: Ảnh nhị phân hoặc grayscale.
        kernel_size: Kích thước phần tử cấu trúc (NxN).
        iterations: Số lần lặp.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def dilate(img: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Giãn nở (Dilation) — phóng to vùng trắng, lấp lỗ nhỏ.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)


def opening(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Mở (Opening) = Erosion → Dilation.
    Loại bỏ nhiễu nhỏ bên ngoài đối tượng.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def closing(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Đóng (Closing) = Dilation → Erosion.
    Lấp các lỗ hổng nhỏ bên trong đối tượng.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def morphological_gradient(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Gradient hình thái = Dilation - Erosion.
    Tìm biên (outline) của đối tượng.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)


def tophat(img: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Top-hat = Ảnh gốc - Opening. Tìm chi tiết sáng nhỏ trên nền sáng."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def blackhat(img: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Black-hat = Closing - Ảnh gốc. Tìm lỗ hổng nhỏ trên nền tối."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def create_clean_mask(
    img: np.ndarray, kernel_size: int = 5
) -> np.ndarray:
    """
    Pipeline tạo mask sạch từ ảnh đầu vào:
    1. Chuyển sang HSV.
    2. Threshold bằng Otsu trên kênh V.
    3. Opening (loại nhiễu nhỏ).
    4. Closing (lấp lỗ hổng).

    Returns:
        Mask nhị phân sạch (0/255).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Otsu thresholding
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Làm sạch bằng morphology
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.05_morphology
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang xử lý ảnh: {img_path}")

    img = load_image(img_path)

    # Tạo mask ban đầu (Otsu)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Các phép morphology
    eroded = erode(mask)
    dilated = dilate(mask)
    opened = opening(mask)
    closed = closing(mask)
    grad = morphological_gradient(mask)
    clean = create_clean_mask(img)

    output_path = str(OUTPUT_DIR / "05_morphology_result.png")
    show_images(
        [mask, eroded, dilated, opened, closed, grad],
        ["Otsu Mask", "Erosion", "Dilation", "Opening", "Closing", "Gradient"],
        save_path=output_path,
        figsize=(20, 4),
    )
    print(f"[DEMO] Kết quả đã lưu tại: {output_path}")
