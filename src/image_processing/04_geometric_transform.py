"""
04_geometric_transform.py
Biến đổi hình học: Xoay, co giãn, Affine, Homography (Perspective).

Lý thuyết (CVIP Chương 2):
- Tịnh tiến, Quay, Tỉ lệ, Biến dạng (Shear).
- Affine: bảo toàn đường thẳng song song (cần 3 cặp điểm).
- Projective/Homography: biến đổi 3D→2D, không bảo toàn song song (cần 4 cặp điểm).
  → Nền tảng của multi-view geometry trong 3D reconstruction.
- Thứ bậc: Translation ⊂ Euclidean ⊂ Similarity ⊂ Affine ⊂ Projective.
"""

import cv2
import numpy as np
from typing import Tuple


def rotate_image(
    img: np.ndarray, angle: float, center: Tuple[int, int] = None, scale: float = 1.0
) -> np.ndarray:
    """
    Xoay ảnh quanh tâm chỉ định.

    Args:
        img: Ảnh đầu vào.
        angle: Góc xoay (độ), dương = ngược chiều kim đồng hồ.
        center: Tâm xoay (x, y). Mặc định = tâm ảnh.
        scale: Hệ số phóng to/thu nhỏ.
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def resize_image(img: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """
    Thay đổi kích thước ảnh.
    Nếu chỉ cho width hoặc height, tự tính theo tỉ lệ.
    """
    h, w = img.shape[:2]

    if width is None and height is None:
        return img

    if width is not None and height is not None:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    if width is not None:
        ratio = width / w
        new_h = int(h * ratio)
        return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_LINEAR)

    # height is not None
    ratio = height / h
    new_w = int(w * ratio)
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_LINEAR)


def affine_transform(
    img: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
) -> np.ndarray:
    """
    Biến đổi Affine — cần 3 cặp điểm tương ứng.
    Bảo toàn: đường thẳng song song, tỉ lệ diện tích.

    Args:
        img: Ảnh đầu vào.
        pts_src: 3 điểm nguồn (3x2 float32).
        pts_dst: 3 điểm đích (3x2 float32).
    """
    h, w = img.shape[:2]
    M = cv2.getAffineTransform(
        np.float32(pts_src), np.float32(pts_dst)
    )
    return cv2.warpAffine(img, M, (w, h))


def perspective_transform(
    img: np.ndarray,
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    output_size: Tuple[int, int] = None,
) -> np.ndarray:
    """
    Biến đổi phối cảnh (Homography) — cần 4 cặp điểm tương ứng.
    Đây là phép biến đổi tổng quát nhất (ma trận 3x3).

    Ứng dụng: rectify ảnh multi-view, ước lượng pose camera.

    Args:
        img: Ảnh đầu vào.
        pts_src: 4 điểm nguồn (4x2 float32).
        pts_dst: 4 điểm đích (4x2 float32).
        output_size: (width, height) ảnh kết quả. Mặc định = kích thước ảnh gốc.
    """
    h, w = img.shape[:2]
    if output_size is None:
        output_size = (w, h)

    M = cv2.getPerspectiveTransform(
        np.float32(pts_src), np.float32(pts_dst)
    )
    return cv2.warpPerspective(img, M, output_size)


def find_homography(
    pts_src: np.ndarray, pts_dst: np.ndarray, method: int = cv2.RANSAC
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ước lượng Homography tự động từ các cặp điểm tương ứng + RANSAC.
    Đây là bước cốt lõi trong Structure-from-Motion.

    Args:
        pts_src: Nx2 array tọa độ điểm nguồn.
        pts_dst: Nx2 array tọa độ điểm đích.

    Returns:
        (H, mask): Ma trận Homography 3x3 và mask inliers.
    """
    H, mask = cv2.findHomography(
        np.float32(pts_src), np.float32(pts_dst), method, 5.0
    )
    return H, mask


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.04_geometric_transform
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang xử lý ảnh: {img_path}")

    img = load_image(img_path)
    h, w = img.shape[:2]

    # Xoay 45 độ
    rotated = rotate_image(img, 45)

    # Affine (3 cặp điểm)
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    affine = affine_transform(img, pts1, pts2)

    # Perspective (4 cặp điểm)
    pts1_p = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2_p = np.float32([[50, 0], [w - 50, 50], [0, h], [w, h - 50]])
    persp = perspective_transform(img, pts1_p, pts2_p)

    output_path = str(OUTPUT_DIR / "04_geometric_transform_result.png")
    show_images(
        [img, rotated, affine, persp],
        ["Ảnh gốc", "Xoay 45°", "Affine Transform", "Perspective Transform"],
        save_path=output_path,
    )
    print(f"[DEMO] Kết quả đã lưu tại: {output_path}")
