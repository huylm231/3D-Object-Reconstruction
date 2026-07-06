"""
06_edge_detection.py
Phát hiện cạnh: Canny, Sobel, Laplacian.

Lý thuyết (CVIP Chương 4):
- Canny: thuật toán phát hiện cạnh nhiều bước (Gaussian blur → gradient →
  non-maximum suppression → hysteresis thresholding).
- Sobel: tính gradient theo hướng x, y bằng kernel 3x3.
- Laplacian: tính đạo hàm bậc 2 — phát hiện cạnh mọi hướng.
- Ứng dụng 3D reconstruction: edge-based segmentation, kiểm tra chất lượng
  feature trước khi matching.
"""

import cv2
import numpy as np


def canny_edge(
    img: np.ndarray, low_threshold: int = 100, high_threshold: int = 200
) -> np.ndarray:
    """
    Canny Edge Detection — thuật toán phát hiện cạnh phổ biến nhất.

    Quy trình:
    1. Gaussian Blur (làm mịn nhiễu).
    2. Tính gradient (Sobel x + y).
    3. Non-Maximum Suppression (giữ điểm cạnh mạnh nhất).
    4. Hysteresis Thresholding (low/high) — nối cạnh yếu liên thông với cạnh mạnh.

    Args:
        low_threshold: Ngưỡng thấp cho hysteresis.
        high_threshold: Ngưỡng cao cho hysteresis.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur trước để giảm nhiễu
    blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.4)
    return cv2.Canny(blurred, low_threshold, high_threshold)


def sobel_edge(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Sobel Edge Detection — tính gradient theo hướng x và y.

    Returns:
        Ảnh biên (gradient magnitude) — kết hợp cả 2 hướng.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gradient theo x và y
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    # Magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(np.clip(magnitude, 0, 255))


def laplacian_edge(img: np.ndarray) -> np.ndarray:
    """
    Laplacian Edge Detection — đạo hàm bậc 2.
    Phát hiện cạnh ở mọi hướng (không cần tính gradient riêng x, y).
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur trước
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    return np.uint8(np.abs(laplacian))


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.06_edge_detection
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang xử lý ảnh: {img_path}")

    img = load_image(img_path)

    canny = canny_edge(img, 100, 200)
    sobel = sobel_edge(img)
    laplacian = laplacian_edge(img)

    output_path = str(OUTPUT_DIR / "06_edge_detection_result.png")
    show_images(
        [img, canny, sobel, laplacian],
        ["Ảnh gốc", "Canny Edge", "Sobel Edge", "Laplacian Edge"],
        save_path=output_path,
    )
    print(f"[DEMO] Kết quả đã lưu tại: {output_path}")
