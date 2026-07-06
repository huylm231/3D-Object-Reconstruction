"""
11_point_cloud.py
Tạo Point Cloud từ depth map + ảnh màu.

Lý thuyết (CVIP Chương 10):
- Depth map + Camera intrinsics K → Point Cloud 3D.
- Mỗi pixel (u, v) với depth Z → tọa độ 3D:
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
- Kết quả: đám mây điểm (point cloud) — tập hợp điểm 3D có tọa độ + màu.
- Xuất định dạng PLY để dùng tiếp cho mesh reconstruction.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def depth_to_pointcloud(
    depth_map: np.ndarray,
    color_img: np.ndarray,
    fx: float = None,
    fy: float = None,
    cx: float = None,
    cy: float = None,
    depth_scale: float = 1000.0,
    depth_trunc: float = 10.0,
):
    """
    Chuyển depth map + ảnh màu thành Point Cloud bằng Open3D.

    Args:
        depth_map: Depth map float [0, 1] hoặc uint16 (mm).
        color_img: Ảnh BGR gốc.
        fx, fy: Tiêu cự camera theo pixel.
                 Mặc định = max(H, W) nếu không biết.
        cx, cy: Tọa độ điểm chính.
                 Mặc định = tâm ảnh.
        depth_scale: Hệ số chia depth (1000 nếu depth tính bằng mm).
        depth_trunc: Depth tối đa (loại bỏ điểm quá xa).

    Returns:
        open3d.geometry.PointCloud
    """
    import open3d as o3d

    h, w = color_img.shape[:2]
    if fx is None:
        fx = float(max(h, w))
    if fy is None:
        fy = fx
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    # Chuẩn bị depth image
    if depth_map.dtype == np.float32 or depth_map.dtype == np.float64:
        # depth float [0, 1] → uint16 mm
        depth_uint16 = (depth_map * depth_scale).astype(np.uint16)
    else:
        depth_uint16 = depth_map.astype(np.uint16)

    # Tạo Open3D images
    depth_o3d = o3d.geometry.Image(depth_uint16)
    color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(color_rgb)

    # Camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(w), int(h), fx, fy, cx, cy
    )

    # Tạo RGBD image (chỉ dùng depth, bỏ qua màu để tập trung vào hình khối)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d,
        intrinsic,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
    )
    # Sơn toàn bộ mây điểm thành màu trắng
    pcd.paint_uniform_color([0.9, 0.9, 0.9])

    # Ước lượng normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    return pcd


def save_pointcloud(pcd, path: str) -> None:
    """Lưu point cloud ra file PLY."""
    # pyrefly: ignore [missing-import]
    import open3d as o3d
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"[SAVE] Đã lưu point cloud: {path}")


def visualize_pointcloud(pcd) -> None:
    """Hiển thị point cloud 3D bằng Open3D (cần GUI)."""
    import open3d as o3d
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud Viewer",
        width=800,
        height=600,
    )


def pointcloud_info(pcd) -> dict:
    """Lấy thông tin cơ bản về point cloud."""
    import numpy as np
    points = np.asarray(pcd.points)
    info = {
        "num_points": len(points),
        "has_normals": pcd.has_normals(),
        "has_colors": pcd.has_colors(),
    }
    if len(points) > 0:
        info["bbox_min"] = points.min(axis=0).tolist()
        info["bbox_max"] = points.max(axis=0).tolist()
    return info


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.11_point_cloud
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import load_image, get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang tạo point cloud cho ảnh: {img_path}")

    try:
        # Bước 1: Ước lượng depth
        import importlib
        _depth = importlib.import_module('src.image_processing.10_depth_estimation')
        depth_float, depth_u8 = _depth.estimate_depth(img_path, encoder="vits")

        # Bước 2: Tạo point cloud
        color_img = load_image(img_path)
        pcd = depth_to_pointcloud(depth_float, color_img)

        # Thông tin
        info = pointcloud_info(pcd)
        print(f"  Point cloud: {info['num_points']} điểm")
        print(f"  Normals: {info['has_normals']}, Colors: {info['has_colors']}")

        # Lưu
        ply_path = str(OUTPUT_DIR / "point_cloud.ply")
        save_pointcloud(pcd, ply_path)
        print(f"[DEMO] Kết quả đã lưu tại: {ply_path}")

    except ImportError:
        print("[DEMO] Cần cài open3d: pip install open3d")
    except Exception as e:
        print(f"[DEMO] Lỗi: {e}")
