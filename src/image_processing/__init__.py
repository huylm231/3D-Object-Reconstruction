"""
image_processing - Thu muc chua cac thuat toan xu ly anh
cho bai toan 3D Object Reconstruction.

Moi module tuong ung voi 1 buoc trong pipeline xu ly anh,
co the chay rieng le hoac goi tu pipeline tong hop.

Luu y: Ten file bat dau bang so (01_, 02_...) nen can dung
importlib de import trong code Python. Pipeline.py da xu ly viec nay.

Cac module:
  01_color_spaces.py       - Chuyen doi khong gian mau
  02_fourier_filtering.py  - Bien doi Fourier & loc tan so
  03_wavelet_denoising.py  - Bien doi Wavelet & khu nhieu
  04_geometric_transform.py- Bien doi hinh hoc
  05_morphology.py         - Xu ly hinh thai hoc
  06_edge_detection.py     - Phat hien canh
  07_segmentation.py       - Phan doan anh
  08_feature_detection.py  - Phat hien dac trung (SIFT/ORB)
  09_feature_matching.py   - Doi sanh dac trung & RANSAC
  10_depth_estimation.py   - Uoc luong depth (Depth-Anything-V2)
  11_point_cloud.py        - Tao Point Cloud
  12_mesh_reconstruction.py- Dung mesh 3D (TripoSR/Poisson)
  pipeline.py              - Pipeline tong hop 12 buoc
  utils.py                 - Ham tien ich chung
"""

from pathlib import Path

# Duong dan goc cua project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = DATA_DIR / "outputs"
SAMPLE_DIR = DATA_DIR / "sample_images"


def _load_module(name: str):
    """Import module bang importlib (ho tro ten file bat dau bang so)."""
    import importlib
    return importlib.import_module(f"src.image_processing.{name}")
