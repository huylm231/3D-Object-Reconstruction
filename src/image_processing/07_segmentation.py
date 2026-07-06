"""
07_segmentation.py
Phân đoạn ảnh: Otsu, K-Means, Watershed, Color-based.

Lý thuyết (CVIP Chương 4):
- Semantic Segmentation: gán nhãn lớp cho từng pixel.
- Otsu: tự động chọn ngưỡng tối ưu dựa trên histogram.
- K-Means: gom cụm pixel theo màu sắc trong không gian RGB/HSV/Lab.
- Watershed: dùng "đường phân thủy" để tách vùng.
- Ứng dụng 3D reconstruction: tách vật thể khỏi nền (object masking),
  tạo silhouette cho Shape-from-Silhouette / Visual Hull.
"""

import cv2
import numpy as np
from typing import Tuple


def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """
    Phân đoạn bằng Otsu's thresholding — tự động chọn ngưỡng.
    Tìm ngưỡng T sao cho phương sai liên lớp (inter-class variance) lớn nhất.

    Returns:
        Mask nhị phân (0/255).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def kmeans_segment(img: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Phân đoạn bằng K-Means clustering trong không gian màu.

    Quy trình:
    1. Reshape ảnh thành (N, 3) — mỗi pixel 1 hàng.
    2. K-Means gom thành K cụm.
    3. Thay mỗi pixel bằng màu tâm cụm tương ứng.

    Args:
        k: Số cụm (số vùng phân đoạn mong muốn).

    Returns:
        Ảnh đã phân đoạn (mỗi vùng 1 màu đồng nhất).
    """
    Z = img.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img.shape)
    return segmented


def watershed_segment(img: np.ndarray) -> np.ndarray:
    """
    Phân đoạn bằng thuật toán Watershed.

    Quy trình:
    1. Otsu → mask nhị phân.
    2. Distance transform → tìm sure foreground (tâm vật thể).
    3. Dilation → sure background.
    4. Unknown = background - foreground.
    5. Connected components → markers.
    6. Watershed → đường phân thủy (markers == -1).

    Returns:
        Ảnh với đường phân thủy được đánh dấu đỏ.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Loại nhiễu bằng opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground (dùng distance transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    result = img.copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = [0, 0, 255]  # Đường phân thủy = đỏ

    return result


def color_based_segment(
    img: np.ndarray,
    lower_hsv: np.ndarray,
    upper_hsv: np.ndarray,
) -> np.ndarray:
    """
    Phân đoạn theo dải màu HSV — tách vật thể có màu cụ thể.

    Args:
        lower_hsv: Giới hạn dưới [H, S, V] (vd: np.array([30, 150, 50])).
        upper_hsv: Giới hạn trên [H, S, V] (vd: np.array([90, 255, 255])).

    Returns:
        Mask nhị phân (vùng trong dải màu = 255).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return mask


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.07_segmentation
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang xử lý ảnh: {img_path}")

    img = load_image(img_path)

    # Các phương pháp phân đoạn
    otsu = otsu_threshold(img)
    kmeans = kmeans_segment(img, k=4)
    watershed = watershed_segment(img)

    output_path = str(OUTPUT_DIR / "07_segmentation_result.png")
    show_images(
        [img, otsu, kmeans, watershed],
        ["Ảnh gốc", "Otsu Threshold", "K-Means (K=4)", "Watershed"],
        save_path=output_path,
    )
    print(f"[DEMO] Kết quả đã lưu tại: {output_path}")
