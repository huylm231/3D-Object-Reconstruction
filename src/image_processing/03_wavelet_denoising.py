"""
03_wavelet_denoising.py
Biến đổi Wavelet & Khử nhiễu ảnh.

Lý thuyết (CVIP Chương 2):
- Wavelet phân tích cả thời gian/không gian và tần số (khác Fourier chỉ tần số).
- 2D-DWT phân tách ảnh thành 4 băng con: LL (xấp xỉ), LH, HL, HH (chi tiết).
- Khử nhiễu: threshold hệ số wavelet chi tiết → tái tạo ảnh sạch.
- Wavelet hash: so sánh tương đồng ảnh để loại ảnh trùng trong tập multi-view.
"""

import cv2
import numpy as np
from typing import Tuple, List


def wavelet_denoise(
    img: np.ndarray, wavelet: str = "db4", level: int = 2
) -> np.ndarray:
    """
    Khử nhiễu ảnh bằng Wavelet (VisuShrink threshold).

    Quy trình:
    1. Phân tách wavelet (wavedec2) — tách ảnh thành LL + các mức chi tiết.
    2. Ước lượng ngưỡng sigma từ hệ số chi tiết mức cao nhất.
    3. Áp dụng soft thresholding — giảm các hệ số nhiễu nhỏ về 0.
    4. Tái tạo ảnh sạch (waverec2).

    Args:
        img: Ảnh grayscale (float hoặc uint8).
        wavelet: Loại wavelet ('db4', 'haar', 'sym4', ...).
        level: Số mức phân tách.

    Returns:
        Ảnh đã khử nhiễu.
    """
    import pywt

    img_float = img.astype(np.float64)
    if len(img_float.shape) == 3:
        img_float = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(
            np.float64
        )

    # Phân tách wavelet
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)

    # Ước lượng ngưỡng VisuShrink: sigma * sqrt(2 * log(N))
    # sigma ước lượng từ median absolute deviation của hệ số chi tiết mức cao nhất
    detail_coeffs = coeffs[-1]  # (LH, HL, HH) của mức cuối cùng
    sigma = np.median(np.abs(detail_coeffs[0])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(img_float.size))

    # Soft thresholding cho tất cả các mức chi tiết
    coeffs_thresh = [coeffs[0]]  # Giữ nguyên LL (xấp xỉ)
    for detail_level in coeffs[1:]:
        coeffs_thresh.append(
            tuple(pywt.threshold(c, threshold, mode="soft") for c in detail_level)
        )

    # Tái tạo ảnh
    denoised = pywt.waverec2(coeffs_thresh, wavelet)
    return np.uint8(np.clip(denoised, 0, 255))


def wavelet_decompose(
    img: np.ndarray, wavelet: str = "db4", level: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Phân tách wavelet 1 mức — trả về 4 băng con: LL, LH, HL, HH.

    - LL: Xấp xỉ (approximation) — phiên bản thu nhỏ mịn của ảnh gốc.
    - LH: Chi tiết ngang (horizontal detail).
    - HL: Chi tiết dọc (vertical detail).
    - HH: Chi tiết chéo (diagonal detail).

    Returns:
        (LL, LH, HL, HH) — mỗi ảnh là numpy array uint8.
    """
    import pywt

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.dwt2(img.astype(np.float64), wavelet)
    LL, (LH, HL, HH) = coeffs

    # Chuẩn hóa về 0-255 để hiển thị
    def normalize(arr):
        arr = np.abs(arr)
        if arr.max() > 0:
            arr = arr / arr.max() * 255
        return np.uint8(arr)

    return normalize(LL), normalize(LH), normalize(HL), normalize(HH)


def wavelet_hash(img: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    Tạo hash wavelet của ảnh — dùng để so sánh tương đồng giữa các ảnh.
    Ứng dụng: loại bỏ ảnh trùng/gần trùng trong tập ảnh multi-view.

    Args:
        img: Ảnh (grayscale hoặc BGR).

    Returns:
        numpy array nhị phân (0/1) — hash của ảnh.
    """
    import pywt

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_float = img.astype(np.float64)
    coeffs = pywt.wavedec2(img_float, wavelet, level=level)

    # Flatten tất cả hệ số
    flat = np.concatenate(
        [np.ravel(coeffs[0])]
        + [np.ravel(d) for band in coeffs[1:] for d in band]
    )

    # So sánh với median → binary hash
    median = np.median(flat)
    return (flat > median).astype(np.int8)


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Khoảng cách Hamming giữa 2 hash (số bit khác nhau)."""
    min_len = min(len(h1), len(h2))
    return int(np.sum(h1[:min_len] != h2[:min_len]))


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.03_wavelet_denoising
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang xử lý ảnh: {img_path}")

    img = load_image(img_path, mode="gray")

    # Thêm nhiễu Gaussian để demo
    noise = np.random.normal(0, 25, img.shape).astype(np.float64)
    noisy = np.uint8(np.clip(img.astype(np.float64) + noise, 0, 255))

    # Khử nhiễu
    denoised = wavelet_denoise(noisy, wavelet="db4", level=2)

    # Phân tách wavelet
    LL, LH, HL, HH = wavelet_decompose(img)

    # So sánh: ảnh gốc vs nhiễu vs denoised
    output1 = str(OUTPUT_DIR / "03_wavelet_denoise_result.png")
    show_images(
        [img, noisy, denoised],
        ["Ảnh gốc", "Ảnh + Nhiễu Gaussian", "Wavelet Denoised"],
        save_path=output1,
    )

    # Phân tách wavelet bands
    output2 = str(OUTPUT_DIR / "03_wavelet_decompose_result.png")
    show_images(
        [LL, LH, HL, HH],
        ["LL (Xấp xỉ)", "LH (Ngang)", "HL (Dọc)", "HH (Chéo)"],
        save_path=output2,
    )
    print(f"[DEMO] Kết quả đã lưu tại: {output1}, {output2}")
