"""
image_processing/utils.py
Các hàm tiện ích dùng chung cho tất cả các module xử lý ảnh.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


def load_image(path: str, mode: str = "color") -> np.ndarray:
    """
    Đọc ảnh từ file.

    Args:
        path: Đường dẫn tới file ảnh.
        mode: 'color' (BGR), 'gray' (grayscale), 'unchanged' (giữ nguyên).

    Returns:
        numpy array chứa dữ liệu ảnh.
    """
    flags = {
        "color": cv2.IMREAD_COLOR,
        "gray": cv2.IMREAD_GRAYSCALE,
        "unchanged": cv2.IMREAD_UNCHANGED,
    }
    img = cv2.imread(str(path), flags.get(mode, cv2.IMREAD_COLOR))
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {path}")
    return img


def save_image(path: str, img: np.ndarray) -> None:
    """Lưu ảnh ra file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    print(f"[SAVE] Đã lưu ảnh: {path}")


def ensure_output_dir(path: str) -> Path:
    """Tạo thư mục output nếu chưa tồn tại."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def show_images(
    images: List[np.ndarray],
    titles: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Hiển thị nhiều ảnh cạnh nhau bằng matplotlib.

    Args:
        images: Danh sách các ảnh (numpy array).
        titles: Danh sách tiêu đề tương ứng.
        save_path: Nếu chỉ định, lưu ảnh kết quả thay vì hiển thị.
        figsize: Kích thước figure (width, height).
    """
    import matplotlib
    matplotlib.use("Agg")  # Backend không cần GUI
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        # Chuyển BGR → RGB nếu ảnh màu
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_show = img

        ax.imshow(img_show, cmap="gray" if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Đã lưu hình so sánh: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Chuyển ảnh từ BGR (OpenCV) sang RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Chuyển ảnh từ RGB sang BGR (OpenCV)."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def get_sample_image_path() -> str:
    """
    Tìm ảnh mẫu để demo.
    Ưu tiên: data/sample_images/ → data/uploads/ → bất kỳ ảnh nào trong data/
    """
    from src.image_processing import SAMPLE_DIR, DATA_DIR

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

    # Tìm trong sample_images
    for ext in extensions:
        files = list(SAMPLE_DIR.glob(ext))
        if files:
            return str(files[0])

    # Tìm trong uploads
    upload_dir = DATA_DIR / "uploads"
    for ext in extensions:
        files = list(upload_dir.glob(ext))
        if files:
            return str(files[0])

    raise FileNotFoundError(
        "Không tìm thấy ảnh mẫu! Hãy đặt ảnh vào data/sample_images/"
    )
