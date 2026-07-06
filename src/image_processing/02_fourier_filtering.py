"""
02_fourier_filtering.py
Biến đổi Fourier & Lọc tần số ảnh.

Lý thuyết (CVIP Chương 2):
- DFT (Discrete Fourier Transform): chuyển ảnh từ miền không gian sang miền tần số.
- FFT (Fast Fourier Transform): thuật toán nhanh tính DFT.
- Tần số thấp = chi tiết lớn, vùng sáng-tối đều.
- Tần số cao = cạnh, chi tiết nhỏ, nhiễu.
- Lọc thông thấp (Low-pass): làm mờ / khử nhiễu.
- Lọc thông cao (High-pass): tăng cường cạnh.
- Ứng dụng 3D reconstruction: khử nhiễu ảnh multi-view trước khi trích đặc trưng,
  giúp SIFT/SURF ổn định hơn.
"""

import cv2
import numpy as np
from typing import Tuple


def apply_fft(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính FFT 2D của ảnh.

    Args:
        img: Ảnh grayscale (H, W).

    Returns:
        (fshift, magnitude_spectrum):
            - fshift: FFT đã shift (tần số 0 ở giữa)
            - magnitude_spectrum: Phổ biên độ (để hiển thị)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Phổ biên độ (log scale để dễ nhìn)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    magnitude = np.uint8(magnitude / magnitude.max() * 255)

    return fshift, magnitude


def lowpass_filter(img: np.ndarray, radius: int = 30) -> np.ndarray:
    """
    Lọc thông thấp bằng Fourier — giữ tần số thấp, loại tần số cao.
    Hiệu ứng: làm mờ / khử nhiễu ảnh.

    Args:
        img: Ảnh grayscale.
        radius: Bán kính vùng lọc (pixel). Nhỏ hơn = mờ hơn.

    Returns:
        Ảnh đã lọc thông thấp.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # FFT
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Tạo mask tròn (giữ tần số thấp ở trung tâm)
    mask = np.zeros((rows, cols), np.float32)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)

    # Áp dụng mask và inverse FFT
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return np.uint8(np.clip(img_back, 0, 255))


def highpass_filter(img: np.ndarray, radius: int = 30) -> np.ndarray:
    """
    Lọc thông cao bằng Fourier — giữ tần số cao, loại tần số thấp.
    Hiệu ứng: tăng cường cạnh / chi tiết.

    Args:
        img: Ảnh grayscale.
        radius: Bán kính vùng loại bỏ. Lớn hơn = nhiều cạnh hơn.

    Returns:
        Ảnh đã lọc thông cao.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # FFT
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)

    # Tạo mask tròn (loại tần số thấp ở trung tâm)
    mask = np.ones((rows, cols), np.float32)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)

    # Áp dụng mask và inverse FFT
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return np.uint8(np.clip(img_back, 0, 255))


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.02_fourier_filtering
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, show_images, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang xử lý ảnh: {img_path}")

    img = load_image(img_path, mode="gray")

    # Phổ tần số
    _, spectrum = apply_fft(img)

    # Lọc
    lp = lowpass_filter(img, radius=30)
    hp = highpass_filter(img, radius=30)

    output_path = str(OUTPUT_DIR / "02_fourier_filtering_result.png")
    show_images(
        [img, spectrum, lp, hp],
        ["Ảnh gốc", "Phổ tần số (FFT)", "Low-pass (r=30)", "High-pass (r=30)"],
        save_path=output_path,
    )
    print(f"[DEMO] Kết quả đã lưu tại: {output_path}")
