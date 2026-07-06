"""
08_feature_detection.py
Phát hiện đặc trưng: SIFT, ORB.

Lý thuyết (CVIP Chương 3):
- Đặc trưng (feature): điểm/vùng đáng chú ý (keypoint/corner) trong ảnh.
- SIFT (Scale-Invariant Feature Transform):
  • DoG (Difference of Gaussian) để tìm cực trị đa scale.
  • Descriptor 128 chiều (4x4x8 histogram gradient).
  • Bất biến scale, rotation, phần ánh sáng.
  • Chậm nhưng chính xác nhất.
- ORB (Oriented FAST and Rotated BRIEF):
  • FAST detector + BRIEF descriptor + oriented.
  • Binary descriptor — rất nhanh.
  • Miễn phí, phù hợp real-time/SLAM.
- Ứng dụng 3D reconstruction: tìm điểm tương ứng giữa các ảnh multi-view,
  làm đầu vào cho SfM / triangulation.
"""

import cv2
import numpy as np
from typing import Tuple, List


def detect_sift(
    img: np.ndarray, n_features: int = 0, mask: np.ndarray = None
) -> Tuple[list, np.ndarray]:
    """
    Phát hiện đặc trưng SIFT.

    Quy trình bên trong:
    1. Lọc Gaussian nhiều scale → tính DoG.
    2. Tìm cực trị scale-space (3x3x3).
    3. Loại điểm kém ổn định (contrast thấp, nằm trên cạnh).
    4. Gán hướng chính → bất biến xoay.
    5. Tạo descriptor 128 chiều.

    Args:
        img: Ảnh (grayscale hoặc BGR).
        n_features: Số đặc trưng tối đa (0 = không giới hạn).
        mask: Mask vùng tìm kiếm (optional).

    Returns:
        (keypoints, descriptors):
            - keypoints: danh sách cv2.KeyPoint
            - descriptors: Nx128 float32 array
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    return keypoints, descriptors


def detect_orb(
    img: np.ndarray, n_features: int = 2000, mask: np.ndarray = None
) -> Tuple[list, np.ndarray]:
    """
    Phát hiện đặc trưng ORB (Oriented FAST + Rotated BRIEF).

    - FAST: phát hiện góc nhanh bằng so sánh pixel vòng tròn.
    - Oriented FAST: thêm hướng (intensity centroid).
    - Rotated BRIEF: binary descriptor, xoay theo hướng keypoint.

    Args:
        n_features: Số đặc trưng tối đa.

    Returns:
        (keypoints, descriptors):
            - keypoints: danh sách cv2.KeyPoint
            - descriptors: Nx32 uint8 array (binary)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    return keypoints, descriptors


def draw_keypoints(
    img: np.ndarray,
    keypoints: list,
    color: Tuple[int, int, int] = (0, 255, 0),
    flags: int = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
) -> np.ndarray:
    """
    Vẽ keypoints lên ảnh.

    Args:
        flags: DRAW_RICH_KEYPOINTS hiển thị cả kích thước + hướng.
    """
    return cv2.drawKeypoints(img, keypoints, None, color=color, flags=flags)


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.08_feature_detection
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang xử lý ảnh: {img_path}")

    img = load_image(img_path)

    # SIFT
    kp_sift, des_sift = detect_sift(img)
    img_sift = draw_keypoints(img, kp_sift, color=(0, 255, 0))
    print(f"  SIFT: {len(kp_sift)} keypoints, descriptor shape: {des_sift.shape if des_sift is not None else 'None'}")

    # ORB
    kp_orb, des_orb = detect_orb(img, n_features=2000)
    img_orb = draw_keypoints(img, kp_orb, color=(255, 0, 0))
    print(f"  ORB:  {len(kp_orb)} keypoints, descriptor shape: {des_orb.shape if des_orb is not None else 'None'}")

    output_path = str(OUTPUT_DIR / "08_feature_detection_result.png")
    show_images(
        [img, img_sift, img_orb],
        [
            "Ảnh gốc",
            f"SIFT ({len(kp_sift)} kp)",
            f"ORB ({len(kp_orb)} kp)",
        ],
        save_path=output_path,
    )
    print(f"[DEMO] Kết quả đã lưu tại: {output_path}")
