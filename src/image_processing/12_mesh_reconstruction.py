"""
12_mesh_reconstruction.py
Dựng mesh 3D: TripoSR (AI), Poisson Surface Reconstruction, Mesh post-processing.

Lý thuyết (CVIP Chương 10):
- TripoSR: Transformer-based, 1 ảnh → mesh 3D hoàn chỉnh (GLB/OBJ).
- Poisson Surface Reconstruction: point cloud + normals → mesh mượt.
- Ball Pivoting Algorithm (BPA): "lăn" quả cầu qua point cloud → tam giác.
- Mesh post-processing:
  • QEM Decimation: giảm số tam giác, giữ hình dạng.
  • Taubin Smoothing: khử nhiễu bề mặt, không co rút.
  • Hole Filling: lấp lỗ hổng.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import warnings

# Ẩn cảnh báo FutureWarning từ thư viện transformers/torch để tránh người dùng nhầm lẫn là lỗi
warnings.filterwarnings("ignore", category=FutureWarning)


def reconstruct_triposr(
    image_path: str,
    output_dir: str,
    mc_resolution: int = 384,
    foreground_ratio: float = 0.85,
) -> str:
    """
    Dựng mesh 3D bằng TripoSR (1 ảnh → mesh GLB).

    Quy trình:
    1. Xoá phông (rembg) — tách vật thể khỏi nền.
    2. Resize foreground — căn giữa vật thể.
    3. TripoSR inference — Transformer dự đoán scene code.
    4. Marching Cubes — trích mesh từ implicit field.
    5. Fix alpha — ép vertex colors thành opaque.
    6. Export GLB.

    Args:
        image_path: Đường dẫn ảnh đầu vào.
        output_dir: Thư mục xuất.
        mc_resolution: Độ phân giải Marching Cubes (cao = chi tiết hơn, chậm hơn).
        foreground_ratio: Tỉ lệ vật thể so với ảnh (0.80 = 80%).

    Returns:
        Đường dẫn file GLB đã tạo.
    """
    import torch
    import rembg
    from PIL import Image
    from src.image_processing import PROJECT_ROOT

    # Thêm src vào sys.path
    src_dir = str(PROJECT_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    glb_path = out_dir / f"{stem}.glb"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Khởi tạo model
    print("[MESH] Đang tải TripoSR model...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to(device)

    # Xóa phông
    print("[MESH] Đang xoá phông ảnh...")
    rembg_session = rembg.new_session()
    input_image = Image.open(image_path)
    image = remove_background(input_image, rembg_session, alpha_matting=True)
    image = resize_foreground(image, foreground_ratio)

    # Lưu ảnh tách nền
    nobg_path = out_dir / f"{stem}_nobg.png"
    image.save(str(nobg_path))

    # Chuẩn bị input cho TripoSR
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
    image_pil = Image.fromarray((image_np * 255.0).astype(np.uint8))

    # Inference
    print("[MESH] Đang dự đoán 3D...")
    with torch.no_grad():
        scene_codes = model([image_pil], device=device)

    # Trích mesh — KHÔNG lấy vertex_colors từ TripoSR (để tránh alpha BLEND gốc)
    print(f"[MESH] Marching Cubes (resolution={mc_resolution})...")
    meshes = model.extract_mesh(scene_codes, False, resolution=mc_resolution)
    mesh = meshes[0]

    import trimesh
    # Repair topology: attempt to make mesh watertight before creating clean plaster mesh
    try:
        from src.image_processing.utils import ensure_watertight
        repaired_mesh, repair_report = ensure_watertight(mesh, hole_fill_method="auto")
        mesh = repaired_mesh
    except Exception as exc:
        print(f"[TOPO] Không thể sửa topology tự động: {exc}")
    
    # Làm mượt bề mặt (Taubin smoothing) — triệt tiêu gợn sóng lồi lõm
    print("[MESH] Đang tối ưu độ sắc nét và làm mượt bề mặt...")
    # TẠM TẮT: Làm mượt làm lệch tọa độ vertices, khiến triplane query màu bị sai!
    # trimesh.smoothing.filter_taubin(mesh, iterations=10)

    # Tạo mesh thạch cao trắng (Fallback / Base)
    clean_mesh = trimesh.Trimesh(
        vertices=np.array(mesh.vertices),
        faces=np.array(mesh.faces),
        process=False,
    )
    # Ensure consistent outward normals for solid plaster look
    try:
        clean_mesh.fix_normals()
    except Exception:
        pass
    # Căn chỉnh hệ tọa độ TripoSR sang chuẩn GLB (Y-up)
    clean_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    clean_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    # Bỏ qua fix_normals() vì thuật toán tự động này hay lật ngược mặt 3D
    # clean_mesh.fix_normals()
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],  # #ffffff
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=True,
    )
    clean_mesh.visual = trimesh.visual.color.ColorVisuals(mesh=clean_mesh)
    clean_mesh.visual.material = material

    # Verify watertight before export
    try:
        from src.image_processing.utils import verify_watertight_solid
        verify_watertight_solid(clean_mesh)
    except Exception as exc:
        print(f"[VERIFY] Mesh không watertight trước khi xuất trắng: {exc}")
        # proceed to export anyway but warn

    # Export GLB (mesh trắng)
    clean_mesh.export(str(glb_path), file_type='glb')
    print(f"[MESH] Đã xuất mesh trắng: {glb_path}")

    return str(glb_path), mesh, model, scene_codes[0]


def poisson_reconstruction(pcd, depth=10):
    """
    Screened Poisson Surface Reconstruction.
    Tạo mesh kín từ Point Cloud, bao quanh điểm như "vải bọc".

    Args:
        pcd: open3d.geometry.PointCloud
        depth: Độ sâu octree (càng cao lưới càng chi tiết nhưng nặng hơn). Khuyến nghị = 10.
    Returns:
        open3d.geometry.TriangleMesh
    """
    import open3d as o3d

    # Đảm bảo có normals
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
    pcd.orient_normals_consistent_tangent_plane(30)

    # Poisson reconstruction
    print(f"  [MESH] Đang nội suy bề mặt bằng Screened Poisson (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=-1
    )

    # Loại bỏ vùng "phồng" có density thấp
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Khử nhiễu bề mặt (Smoothing) để làm lưới mượt hơn
    mesh = mesh.filter_smooth_simple(number_of_iterations=3)

    return mesh


def ball_pivoting_reconstruction(pcd):
    """
    Ball Pivoting Algorithm (BPA) — "lăn" quả cầu qua point cloud → tam giác.
    Giữ chi tiết gốc tốt hơn Poisson.

    Returns:
        open3d.geometry.TriangleMesh
    """
    import open3d as o3d

    if not pcd.has_normals():
        pcd.estimate_normals()

    # Tính bán kính dựa trên khoảng cách trung bình
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radii = [avg_dist, avg_dist * 2, avg_dist * 4]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh


def simplify_mesh(mesh, target_triangles: int = 20000):
    """
    Giảm số tam giác bằng Quadric Error Metrics (QEM) decimation.
    Gộp cạnh gây ít sai số hình học nhất.

    Args:
        target_triangles: Số tam giác mong muốn.
    """
    return mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )


def smooth_mesh(mesh, iterations: int = 20):
    """
    Taubin smoothing — khử nhiễu bề mặt mà không co rút mesh.
    Xen kẽ bước co (λ) và giãn (μ).
    """
    mesh_smooth = mesh.filter_smooth_taubin(number_of_iterations=iterations)
    mesh_smooth.compute_vertex_normals()
    return mesh_smooth


def save_mesh(mesh, path: str) -> None:
    """Lưu mesh ra file (PLY/OBJ/GLB)."""
    import open3d as o3d
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)
    print(f"[SAVE] Đã lưu mesh: {path}")


# ═══════════════════════════════════════════════════════════════
#  DEMO — Chạy: python -m src.image_processing.12_mesh_reconstruction
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from src.image_processing.utils import get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    img_path = get_sample_image_path()
    print(f"[DEMO] Đang dựng mesh 3D cho ảnh: {img_path}")

    try:
        # [B3] reconstruct_triposr() tr\u1ea3 tuple 4 ph\u1ea7n t\u1eed: (glb_path, mesh, model, scene_code)
        glb_path, _, _, _ = reconstruct_triposr(
            img_path,
            str(OUTPUT_DIR),
            mc_resolution=384,
        )
        print(f"[DEMO] Mesh GLB \u0111\u00e3 t\u1ea1o: {glb_path}")

    except Exception as e:
        print(f"[DEMO] Lỗi TripoSR: {e}")
        print("[DEMO] Thử Poisson reconstruction từ point cloud...")

        try:
            import importlib
            _depth = importlib.import_module('src.image_processing.10_depth_estimation')
            _pc = importlib.import_module('src.image_processing.11_point_cloud')
            from src.image_processing.utils import load_image

            depth_float, _ = _depth.estimate_depth(img_path)
            color_img = load_image(img_path)
            pcd = _pc.depth_to_pointcloud(depth_float, color_img)

            mesh = poisson_reconstruction(pcd, depth=8)
            mesh = smooth_mesh(mesh)

            mesh_path = str(OUTPUT_DIR / "mesh_poisson.ply")
            save_mesh(mesh, mesh_path)
            print(f"[DEMO] Poisson mesh đã tạo: {mesh_path}")

        except Exception as e2:
            print(f"[DEMO] Lỗi Poisson: {e2}")
