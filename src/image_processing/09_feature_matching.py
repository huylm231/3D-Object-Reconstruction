"""
09_feature_matching.py
Đối sánh đặc trưng: FLANN, BFMatcher, Lowe's ratio test, RANSAC, Homography.

Lý thuyết (CVIP Chương 3):
- Đối sánh descriptor giữa 2 ảnh: Brute-Force (so hết) hoặc FLANN (tìm nhanh).
- Lowe's ratio test: loại match có distance(best)/distance(second) > 0.75.
- RANSAC (Random Sample Consensus): loại outlier khi ước lượng Homography/Essential Matrix.
- Ứng dụng 3D reconstruction: tìm cặp điểm tương ứng giữa multi-view images,
  làm đầu vào cho ước lượng pose camera (Essential Matrix → R, t).
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def match_sift(
    img1: np.ndarray,
    img2: np.ndarray,
    ratio_threshold: float = 0.75,
    mask1: np.ndarray = None,
    mask2: np.ndarray = None,
) -> Tuple[list, np.ndarray, list, np.ndarray, list]:
    """
    Đối sánh SIFT giữa 2 ảnh bằng FLANN + Lowe's ratio test.

    Quy trình:
    1. Detect SIFT trên cả 2 ảnh.
    2. FLANN kNN matcher (k=2) — tìm 2 match gần nhất cho mỗi descriptor.
    3. Lowe's ratio test: giữ match nếu d(best) < ratio * d(second).

    Returns:
        (kp1, des1, kp2, des2, good_matches)
    """
    import importlib
    _fd = importlib.import_module('src.image_processing.08_feature_detection')
    detect_sift = _fd.detect_sift

    kp1, des1 = detect_sift(img1, mask=mask1)
    kp2, des2 = detect_sift(img2, mask=mask2)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return kp1, des1, kp2, des2, []

    # FLANN matcher (phù hợp descriptor dạng float của SIFT)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return kp1, des1, kp2, des2, good_matches


def match_orb(
    img1: np.ndarray,
    img2: np.ndarray,
    n_features: int = 2000,
    mask1: np.ndarray = None,
    mask2: np.ndarray = None,
) -> Tuple[list, np.ndarray, list, np.ndarray, list]:
    """
    Đối sánh ORB giữa 2 ảnh bằng BFMatcher + cross-check.

    ORB dùng binary descriptor → dùng Hamming distance (đếm bit khác nhau).

    Returns:
        (kp1, des1, kp2, des2, matches_sorted)
    """
    import importlib
    _fd = importlib.import_module('src.image_processing.08_feature_detection')
    detect_orb = _fd.detect_orb

    kp1, des1 = detect_orb(img1, n_features=n_features, mask=mask1)
    kp2, des2 = detect_orb(img2, n_features=n_features, mask=mask2)

    if des1 is None or des2 is None:
        return kp1, des1, kp2, des2, []

    # BFMatcher với Hamming distance + cross-check
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches_sorted = sorted(matches, key=lambda x: x.distance)

    return kp1, des1, kp2, des2, matches_sorted


def find_homography_ransac(
    kp1: list,
    kp2: list,
    matches: list,
    reproj_threshold: float = 5.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Ước lượng Homography bằng RANSAC từ kết quả matching.

    RANSAC:
    1. Chọn ngẫu nhiên 4 cặp điểm (minimum cho homography).
    2. Tính Homography từ 4 cặp đó.
    3. Đếm inliers (điểm có reprojection error < threshold).
    4. Lặp lại N lần, giữ Homography có nhiều inliers nhất.

    Returns:
        (H, mask, src_pts, dst_pts):
            - H: Ma trận Homography 3x3 (hoặc None nếu thất bại).
            - mask: Mask inliers (1 = inlier, 0 = outlier).
            - src_pts, dst_pts: Tọa độ điểm tương ứng.
    """
    if len(matches) < 4:
        return None, None, np.array([]), np.array([])

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold)
    return H, mask, src_pts, dst_pts


def estimate_essential_matrix(
    kp1: list,
    kp2: list,
    matches: list,
    K: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Ước lượng Essential Matrix → Decompose ra (R, t) — pose tương đối giữa 2 camera.
    Đây là bước CỐT LÕI trong Structure-from-Motion (SfM).

    Essential Matrix E = K2^T * F * K1 = [t]_x * R

    Args:
        K: Ma trận camera intrinsics 3x3.

    Returns:
        (R, t, mask): Rotation matrix, translation vector, inlier mask.
    """
    if len(matches) < 5:
        return None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask_e = cv2.findEssentialMat(
        src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )

    if E is None:
        return None, None, None

    _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K)
    return R, t, mask_pose


def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: list,
    kp2: list,
    matches: list,
    max_matches: int = 50,
) -> np.ndarray:
    """Vẽ kết quả matching giữa 2 ảnh."""
    return cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.09_feature_matching
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, save_image
    from src.image_processing import OUTPUT_DIR, SAMPLE_DIR

    # Tìm 2 ảnh để demo matching
    import glob
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    all_images = []
    for ext in extensions:
        all_images.extend(glob.glob(str(SAMPLE_DIR / ext)))

    if len(all_images) < 2:
        # Nếu chỉ có 1 ảnh, tạo ảnh thứ 2 bằng cách xoay
        from src.image_processing.utils import get_sample_image_path
        img_path = get_sample_image_path()
        img1 = load_image(img_path)
        import importlib
        _gt = importlib.import_module('src.image_processing.04_geometric_transform')
        img2 = _gt.rotate_image(img1, 15)
        print(f"[DEMO] Chỉ có 1 ảnh mẫu → tạo ảnh thứ 2 bằng cách xoay 15°")
    else:
        img1 = load_image(all_images[0])
        img2 = load_image(all_images[1])
        print(f"[DEMO] Matching giữa: {all_images[0]} và {all_images[1]}")

    # SIFT matching
    kp1, _, kp2, _, good = match_sift(img1, img2)
    print(f"  SIFT matches: {len(good)} good matches")

    if len(good) >= 4:
        H, mask, _, _ = find_homography_ransac(kp1, kp2, good)
        if mask is not None:
            inliers = int(mask.sum())
            print(f"  Homography inliers: {inliers}/{len(good)}")

    match_img = draw_matches(img1, img2, kp1, kp2, good, max_matches=50)

    output_path = str(OUTPUT_DIR / "09_feature_matching_result.png")
    save_image(output_path, match_img)
    print(f"[DEMO] Kết quả đã lưu tại: {output_path}")
