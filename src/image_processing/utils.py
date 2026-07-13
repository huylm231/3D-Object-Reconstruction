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


def verify_opaque_material(glb_path: str) -> None:
    """Kiểm tra file GLB đã xuất có material alphaMode OPAQUE hay không."""
    try:
        from pygltflib import GLTF2
    except ImportError as exc:
        raise ImportError(
            "pygltflib cần được cài đặt để xác thực alphaMode dalam GLB: pip install pygltflib"
        ) from exc

    gltf = GLTF2().load(str(glb_path))
    if not gltf.materials:
        print(f"[VERIFY] {glb_path} không có materials; theo spec glTF mặc định là OPAQUE")
        return

    for idx, material in enumerate(gltf.materials):
        alpha_mode = material.alphaMode if material.alphaMode is not None else "OPAQUE"
        print(f"[VERIFY] material[{idx}] alphaMode={alpha_mode}")
        if alpha_mode.upper() in {"BLEND", "MASK"}:
            raise RuntimeError(
                f"[VERIFY] {glb_path} chứa material[{idx}] với alphaMode={alpha_mode}. "
                "Phải là OPAQUE."
            )


def _get_boundary_edges(tmesh) -> np.ndarray:
    """Return boundary edges for a trimesh mesh using the available trimesh API."""
    try:
        import trimesh
    except Exception:
        return np.zeros((0, 2), dtype=np.int64)

    try:
        if hasattr(tmesh, "edges_sorted") and hasattr(trimesh.grouping, "group_rows"):
            groups = trimesh.grouping.group_rows(tmesh.edges_sorted, require_count=1)
            if len(groups):
                return tmesh.edges_sorted[groups]
            return np.zeros((0, 2), dtype=np.int64)
    except Exception:
        pass

    faces = np.asarray(getattr(tmesh, "faces", []), dtype=np.int64)
    if faces.size == 0 or faces.ndim != 2 or faces.shape[1] != 3:
        return np.zeros((0, 2), dtype=np.int64)

    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique_edges[counts == 1]
    return boundary.astype(np.int64, copy=False)


def diagnose_mesh_topology(mesh) -> dict:
    """Diagnostic report for mesh topology using trimesh.

    Returns a dict with keys: is_watertight, body_count, boundary_loops (list of dicts with
    vertex_count, centroid, bbox).
    """
    try:
        import trimesh
    except Exception:
        raise ImportError("trimesh cần để chẩn đoán topology của mesh")

    tmesh = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    report = {
        "is_watertight": bool(getattr(tmesh, "is_watertight", False)),
        "body_count": int(getattr(tmesh, "body_count", 1)),
        "boundary_loops": [],
        "boundary_loop_count": 0,
        "boundary_edge_count": 0,
    }

    boundary_edges = _get_boundary_edges(tmesh)
    report["boundary_edge_count"] = int(len(boundary_edges))

    if boundary_edges.size:
        adj = {}
        for a, b in boundary_edges:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))

        visited = set()
        loops = []
        max_loop_length = min(100000, max(1000, len(adj) * 10))
        max_loops = 5000
        for v0 in adj.keys():
            if v0 in visited:
                continue
            loop = [v0]
            visited.add(v0)
            cur = v0
            prev = None
            while len(loop) < max_loop_length:
                nbrs = [n for n in adj.get(cur, []) if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                if nxt == v0:
                    break
                loop.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
            loops.append(loop)
            if len(loops) >= max_loops:
                break

        report["boundary_loop_count"] = int(len(loops))
        if len(loops) <= max_loops:
            for loop in loops:
                if not loop:
                    continue
                pts = np.asarray(tmesh.vertices)[loop]
                centroid = pts.mean(axis=0).tolist()
                bbox = {
                    "min": pts.min(axis=0).tolist(),
                    "max": pts.max(axis=0).tolist(),
                }
                report["boundary_loops"].append({
                    "vertex_count": len(loop),
                    "centroid": centroid,
                    "bbox": bbox,
                })
        else:
            report["boundary_loops"] = []

    return report


def verify_watertight_solid(mesh) -> None:
    """Raise RuntimeError if mesh is not watertight/solid.

    This is a simple guard invoked before final export to avoid exporting open shells.
    """
    try:
        import trimesh
    except Exception:
        raise ImportError("trimesh cần để kiểm tra watertight property")

    tmesh = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    if not getattr(tmesh, "is_watertight", False):
        raise RuntimeError("Mesh chưa watertight sau bước lấp lỗ. Từ chối xuất để tránh hiệu ứng trong suốt.")


def ensure_watertight(
    mesh,
    hole_fill_method: str = "auto",
    voxel_pitch: Optional[float] = None,
    height_axis: Optional[str] = None,
    drop_disconnected_debris: bool = True,
):
    """Attempt to make the mesh watertight.

    Returns: (repaired_mesh, diagnostics)
    """
    try:
        import trimesh
    except Exception:
        raise ImportError("trimesh cần để sửa lỗ hổng mesh")

    tmesh = mesh if isinstance(mesh, trimesh.Trimesh) else trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )

    diag_before = diagnose_mesh_topology(tmesh)
    print(
        f"[TOPO] Trước sửa: is_watertight={diag_before['is_watertight']} body_count={diag_before['body_count']} boundary_loops={len(diag_before['boundary_loops'])}"
    )

    if drop_disconnected_debris:
        try:
            components = tmesh.split(only_watertight=False)
        except Exception:
            components = [tmesh]
        if len(components) > 1:
            largest_idx = max(
                range(len(components)),
                key=lambda i: (len(components[i].vertices), len(components[i].faces)),
            )
            largest_component = components[largest_idx]
            discarded_vertices = sum(len(c.vertices) for i, c in enumerate(components) if i != largest_idx)
            discarded_faces = sum(len(c.faces) for i, c in enumerate(components) if i != largest_idx)
            print(
                f"[TOPO] Phát hiện {len(components)} component(s) không liên thông; giữ component lớn nhất "
                f"({len(largest_component.vertices)} verts, {len(largest_component.faces)} faces), bỏ {discarded_vertices} verts / {discarded_faces} faces"
            )
            tmesh = largest_component

    methods_tried = []

    def try_simple_fill(m):
        try:
            methods_tried.append("simple_fill")
            trimesh.repair.fill_holes(m)
            return getattr(m, "is_watertight", False)
        except Exception:
            return False

    def get_ordered_boundary_loops(m):
        edges = _get_boundary_edges(m)
        if edges.size == 0:
            return []
        adj = {}
        for a, b in edges:
            adj.setdefault(int(a), []).append(int(b))
            adj.setdefault(int(b), []).append(int(a))
        loops = []
        visited = set()
        for start in adj:
            if start in visited:
                continue
            cur = start
            prev = None
            loop = [cur]
            visited.add(cur)
            while True:
                nbrs = [n for n in adj.get(cur, []) if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                if nxt == start:
                    break
                loop.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
            loops.append(loop)
        loops.sort(key=lambda L: -len(L))
        return loops

    success = False

    if hole_fill_method in ("auto", "simple"):
        if try_simple_fill(tmesh):
            success = True

    if not success and hole_fill_method in ("auto", "loop_cap"):
        loops = get_ordered_boundary_loops(tmesh)
        if loops:
            all_new_faces = []
            for chosen in loops:
                if len(chosen) < 3:
                    continue

                pts3 = np.asarray(tmesh.vertices)[chosen]
                centered = pts3 - pts3.mean(axis=0)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                e1 = vh[0]
                e2 = vh[1]
                pts2d = np.dot(centered, np.vstack([e1, e2]).T)

                area2 = 0.0
                for i in range(len(pts2d)):
                    x1, y1 = pts2d[i]
                    x2, y2 = pts2d[(i + 1) % len(pts2d)]
                    area2 += (x1 * y2 - x2 * y1)
                if area2 < 0:
                    pts2d = pts2d[::-1]
                    chosen = chosen[::-1]

                def is_convex(a, b, c):
                    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 1e-8

                def point_in_triangle(pt, a, b, c):
                    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                    if abs(denom) < 1e-12:
                        return False
                    w1 = ((b[1] - c[1]) * (pt[0] - c[0]) + (c[0] - b[0]) * (pt[1] - c[1])) / denom
                    w2 = ((c[1] - a[1]) * (pt[0] - c[0]) + (a[0] - c[0]) * (pt[1] - c[1])) / denom
                    w3 = 1 - w1 - w2
                    return (w1 >= -1e-8) and (w2 >= -1e-8) and (w3 >= -1e-8)

                P = pts2d.tolist()
                idxs = list(range(len(P)))
                new_tris = []
                guard = 0
                while len(idxs) > 3 and guard < 10000:
                    guard += 1
                    found = False
                    for i in range(len(idxs)):
                        i_prev = idxs[i - 1]
                        i_curr = idxs[i]
                        i_next = idxs[(i + 1) % len(idxs)]
                        a = P[i_prev]
                        b = P[i_curr]
                        c = P[i_next]
                        if not is_convex(a, b, c):
                            continue
                        ear_ok = True
                        for j in idxs:
                            if j in (i_prev, i_curr, i_next):
                                continue
                            if point_in_triangle(P[j], a, b, c):
                                ear_ok = False
                                break
                        if not ear_ok:
                            continue
                        new_tris.append((chosen[i_prev], chosen[i_curr], chosen[i_next]))
                        idxs.pop(i)
                        found = True
                        break
                    if not found:
                        break

                if len(idxs) == 3:
                    new_tris.append((chosen[idxs[0]], chosen[idxs[1]], chosen[idxs[2]]))

                if new_tris:
                    all_new_faces.extend(new_tris)

            if all_new_faces:
                methods_tried.append("loop_cap")
                faces_arr = np.asarray(tmesh.faces)
                new_faces = np.vstack([faces_arr, np.asarray(all_new_faces, dtype=np.int64)])
                tmesh = trimesh.Trimesh(vertices=np.asarray(tmesh.vertices), faces=new_faces, process=False)
                if getattr(tmesh, "is_watertight", False):
                    success = True

    if not success and hole_fill_method in ("auto", "voxelize"):
        methods_tried.append("voxelize")
        pitch = voxel_pitch or max(0.003, float(max(np.ptp(np.asarray(tmesh.vertices), axis=0))) / 128.0)
        try:
            try:
                v = tmesh.voxelized(pitch)
                m2 = v.marching_cubes
                if isinstance(m2, trimesh.Trimesh) and getattr(m2, "is_watertight", False):
                    tmesh = m2
                    success = True
            except Exception:
                import open3d as o3d
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tmesh.vertices))
                o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tmesh.faces))
                voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(o3d_mesh, voxel_size=pitch)
                try:
                    vg = voxel_grid
                    pts = []
                    for v in vg.get_voxels():
                        pts.append(v.grid_index)
                    if pts:
                        pts = np.asarray(pts)
                        cubes = []
                        for g in pts:
                            cx = vg.origin + g * pitch
                            box = trimesh.creation.box(
                                extents=(pitch, pitch, pitch),
                                transform=trimesh.transformations.translation_matrix(cx),
                            )
                            cubes.append(box)
                        if cubes:
                            merged = trimesh.util.concatenate(cubes)
                            tmesh = merged.simplify_quadric_decimation(max(100000, len(merged.faces)))
                            if getattr(tmesh, "is_watertight", False):
                                success = True
                except Exception:
                    pass
        except Exception:
            pass

    diag_after = diagnose_mesh_topology(tmesh)
    print(
        f"[TOPO] Sau sửa: is_watertight={diag_after['is_watertight']} body_count={diag_after['body_count']} boundary_loops={len(diag_after['boundary_loops'])} methods_tried={methods_tried}"
    )

    return tmesh, {"before": diag_before, "after": diag_after, "methods_tried": methods_tried}


def render_verification_views(mesh, out_dir: str, stem: str = "verify", size: int = 512) -> dict:
    """Render the mesh from several canonical views and save images.

    This is best-effort and now fails gracefully when offscreen rendering is unsupported.
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {"images": [], "errors": []}

    try:
        import numpy as np
        import open3d as o3d
    except Exception:
        print("[VERIFY] Bỏ qua render xác minh: môi trường không hỗ trợ offscreen rendering")
        return results

    try:
        tri = o3d.geometry.TriangleMesh()
        tri.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        tri.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        tri.compute_vertex_normals()

        renderer = None
        try:
            renderer = o3d.visualization.rendering.OffscreenRenderer(size, size)
        except Exception:
            print("[VERIFY] Bỏ qua render xác minh: môi trường không hỗ trợ offscreen rendering")
            return results

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", tri, mat)

        verts = np.asarray(tri.vertices)
        center = verts.mean(axis=0)
        ext = verts.max(axis=0) - verts.min(axis=0)
        radius = float(np.linalg.norm(ext)) * 1.2

        views = [
            (center + np.array([0.0, 0.0, radius]), [0, 1, 0]),
            (center + np.array([0.0, 0.0, -radius]), [0, 1, 0]),
            (center + np.array([radius, 0.0, 0.0]), [0, 1, 0]),
            (center + np.array([0.0, -radius, 0.0]), [0, 0, 1]),
        ]

        for i, (eye, up) in enumerate(views):
            try:
                renderer.scene.camera.look_at(center, eye, up)
                img = renderer.render_to_image()
                path = out / f"{stem}_view{i}.png"
                o3d.io.write_image(str(path), img)
                results["images"].append(str(path))
            except Exception as exc:
                results["errors"].append(str(exc))

        try:
            renderer.scene.clear_geometry()
            renderer.shutdown()
        except Exception:
            pass
        return results

    except Exception as exc:
        print(f"[VERIFY] Open3D render thất bại: {exc} — thử fallback trimesh...")
        # [B2] Fallback trimesh: trimesh.geometry.look_at() KHÔNG TỒN TẠI trong trimesh
        # → dùng cách đặt camera đơn giản hơn qua scene.camera_transform nếu có,
        # hoặc trả về kết quả rỗng với thông báo rõ ràng thay vì crash.
        try:
            import trimesh as _trimesh
            scene = mesh.scene() if hasattr(mesh, "scene") else None
            if scene is None:
                results["errors"].append("trimesh fallback: mesh.scene() không khả dụng")
                return results

            bbox = np.asarray(mesh.bounds) if hasattr(mesh, "bounds") else None
            if bbox is None:
                results["errors"].append("trimesh fallback: mesh.bounds không khả dụng")
                return results

            center = bbox.mean(axis=0)
            ext = bbox[1] - bbox[0]
            radius = float(np.linalg.norm(ext)) * 1.5

            eye_positions = [
                center + np.array([0.0, 0.0, radius]),
                center + np.array([0.0, 0.0, -radius]),
                center + np.array([radius, 0.0, 0.0]),
                center + np.array([0.0, -radius, 0.0]),
            ]

            for i, eye in enumerate(eye_positions):
                try:
                    # Dùng trimesh camera_transform thay vì look_at() không tồn tại
                    direction = center - eye
                    dist = np.linalg.norm(direction)
                    if dist < 1e-6:
                        continue
                    scene.camera.z_far = dist * 3
                    scene.camera.z_near = dist * 0.01
                    # trimesh.scene.cameras.Camera.look_at tồn tại nhưng gán qua transform
                    forward = direction / dist
                    up = np.array([0.0, 1.0, 0.0])
                    if abs(np.dot(forward, up)) > 0.99:
                        up = np.array([0.0, 0.0, 1.0])
                    right = np.cross(forward, up)
                    right /= np.linalg.norm(right)
                    up = np.cross(right, forward)
                    T = np.eye(4)
                    T[:3, 0] = right
                    T[:3, 1] = up
                    T[:3, 2] = -forward
                    T[:3, 3] = eye
                    scene.camera_transform = T

                    png = scene.save_image(resolution=(size, size))
                    if png:
                        path = out / f"{stem}_view{i}.png"
                        with open(path, 'wb') as f:
                            f.write(png)
                        results["images"].append(str(path))
                except Exception as exc2:
                    results["errors"].append(f"trimesh view {i}: {exc2}")
            return results
        except Exception as exc3:
            results["errors"].append(f"trimesh fallback failed: {exc3}")
            print(f"[VERIFY] trimesh fallback thất bại: {exc3}")

    return results



