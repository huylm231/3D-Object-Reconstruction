import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import trimesh
except ImportError:  # pragma: no cover - environment fallback
    trimesh = None


def _as_trimesh(mesh: Any) -> Any:
    """Chuyển mesh đầu vào về dạng trimesh.Trimesh để thao tác thống nhất."""
    if trimesh is None:
        raise ImportError("trimesh chưa được cài đặt")

    if isinstance(mesh, trimesh.Trimesh):
        return mesh

    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise ValueError("Scene mesh rỗng")
        return trimesh.util.concatenate(geometries)

    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            process=False,
        )

    raise TypeError(f"Không hỗ trợ loại mesh: {type(mesh)}")


def _normalize_rgba(color: Any) -> np.ndarray:
    """Chuyển màu đầu vào về mảng RGBA uint8 4 phần tử."""
    if color is None:
        color = (200, 30, 30, 255)
    if isinstance(color, str):
        color = color.strip()
        if color.startswith("#"):
            color = color[1:]
            if len(color) == 6:
                color = color + "FF"
            if len(color) != 8:
                raise ValueError(f"Hex màu không hợp lệ: {color}")
            return np.array([int(color[i : i + 2], 16) for i in (0, 2, 4, 6)], dtype=np.uint8)
        raise ValueError("Chuỗi màu chỉ hỗ trợ định dạng hex như #RRGGBBAA")

    values = list(color)
    if len(values) == 3:
        values.append(255)
    if len(values) != 4:
        raise ValueError("Màu phải có 3 hoặc 4 thành phần")
    return np.asarray(values, dtype=np.uint8)


def _load_reference_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh tham chiếu: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Không thể đọc ảnh tham chiếu: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3] > 0
        rgb = image[:, :, :3]
    else:
        rgb = image[:, :, :3]
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alpha = alpha > 0
        if not np.any(alpha):
            border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
            bg = np.median(border.astype(np.float32), axis=0)
            dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=2)
            alpha = dist > 20.0
            if not np.any(alpha):
                alpha = np.ones(rgb.shape[:2], dtype=bool)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx((alpha.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return rgb, mask.astype(bool)


def _silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError("Kích thước silhouette phải giống nhau")
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def _render_projected_silhouette(
    mesh: Any,
    projected: np.ndarray,
    valid: np.ndarray,
    image_shape: Tuple[int, int],
    max_faces: int = 3000,
) -> np.ndarray:
    h, w = image_shape
    silhouette = np.zeros((h, w), dtype=np.uint8)
    faces = mesh.faces
    if len(faces) > max_faces:
        indices = np.random.default_rng(0).choice(len(faces), size=max_faces, replace=False)
    else:
        indices = np.arange(len(faces), dtype=np.int32)

    for face_idx in indices:
        pts = projected[faces[face_idx]]
        face_valid = valid[faces[face_idx]]
        if not np.any(face_valid):
            continue
        pts_i = np.round(pts).astype(np.int32)
        pts_i[:, 0] = np.clip(pts_i[:, 0], 0, w - 1)
        pts_i[:, 1] = np.clip(pts_i[:, 1], 0, h - 1)
        if np.linalg.matrix_rank(pts_i.astype(np.float32) - pts_i[0].astype(np.float32)) < 2:
            continue
        cv2.fillConvexPoly(silhouette, pts_i, 255)
    return silhouette > 0


def align_camera_to_reference(
    mesh: Any,
    reference_image_path: str,
    camera_params: Optional[Dict[str, Any]] = None,
    alignment_mode: str = "auto",
    manual_params: Optional[Dict[str, Any]] = None,
    cam_dists: Optional[Tuple[float, ...]] = None,
    fovy_degs: Optional[Tuple[float, ...]] = None,
) -> Dict[str, Any]:
    if camera_params is not None:
        print("[CAMERA] Sử dụng camera_params đã cho sẵn")
        return camera_params

    if alignment_mode == "manual":
        if manual_params is None:
            raise ValueError("manual_params phải được cung cấp khi alignment_mode='manual'")
        print("[CAMERA] Sử dụng manual camera params")
        return manual_params

    if alignment_mode != "auto":
        raise ValueError("alignment_mode phải là 'auto' hoặc 'manual'")

    rgb, mask = _load_reference_image(reference_image_path)
    silhouette_ref = mask
    h, w = rgb.shape[:2]
    vertices = np.asarray(_as_trimesh(mesh).vertices, dtype=np.float32)
    faces = np.asarray(_as_trimesh(mesh).faces, dtype=np.int64)

    if cam_dists is None:
        cam_dists = (1.0, 1.3, 1.6, 1.9, 2.2)
    if fovy_degs is None:
        fovy_degs = (30.0, 35.0, 40.0, 45.0, 50.0)

    best_score = -1.0
    best_params: Dict[str, Any] = {}
    perms = list(itertools.permutations((0, 1, 2)))
    signs = list(itertools.product((1, -1), repeat=3))

    for axis_map in perms:
        for sign in signs:
            for cam_dist in cam_dists:
                for fovy_deg in fovy_degs:
                    tan_half = math.tan(math.radians(fovy_deg) / 2.0)
                    if tan_half <= 0:
                        continue
                    fx = 0.5 * w / tan_half
                    fy = 0.5 * h / tan_half
                    x_cam = vertices[:, axis_map[0]] * sign[0]
                    y_cam = vertices[:, axis_map[1]] * sign[1]
                    z_cam = vertices[:, axis_map[2]] * sign[2] - cam_dist
                    valid = z_cam < 0
                    if np.sum(valid) < max(10, len(vertices) * 0.05):
                        continue
                    u = (w * 0.5) + fx * (x_cam / -z_cam)
                    v = (h * 0.5) + fy * (y_cam / -z_cam)
                    projected = np.stack([u, v], axis=1)
                    silhouette_pred = _render_projected_silhouette(
                        mesh, projected, valid, (h, w)
                    )
                    score = _silhouette_iou(silhouette_ref, silhouette_pred)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "image_size": (w, h),
                            "fovy_deg": float(fovy_deg),
                            "cam_dist": float(cam_dist),
                            "axis_map": tuple(axis_map),
                            "signs": tuple(sign),
                            "cx": float(w * 0.5),
                            "cy": float(h * 0.5),
                            "fx": float(fx),
                            "fy": float(fy),
                        }
    if not best_params:
        raise RuntimeError("Không tìm được camera alignment phù hợp")

    print(f"[CAMERA] Chọn camera auto: fovy={best_params['fovy_deg']} cam_dist={best_params['cam_dist']} axis={best_params['axis_map']} signs={best_params['signs']} score={best_score:.4f}")
    return best_params


def _sample_region_from_mask(
    project_xy: np.ndarray,
    region_mask: np.ndarray,
    default_region: int = 0,
) -> int:
    h, w = region_mask.shape
    u = int(round(project_xy[0]))
    v = int(round(project_xy[1]))
    if 0 <= v < h and 0 <= u < w:
        region_id = int(region_mask[v, u])
        if region_id >= 0:
            return region_id
    return default_region


def extract_color_regions(
    reference_image_path: str,
    cluster_count: int = 3,
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int]], Dict[int, str]]:
    rgb, mask = _load_reference_image(reference_image_path)
    if not np.any(mask):
        raise ValueError("Không thể xác định vật thể trong ảnh tham chiếu")

    h, w = rgb.shape[:2]
    cluster_count = max(2, min(cluster_count, 4))
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    indices = np.where(mask)
    pixels = lab[indices].astype(np.float32)
    y_coords = indices[0].astype(np.float32) / max(1, h - 1)
    features = np.column_stack([pixels, y_coords[:, None] * 60.0])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        features,
        cluster_count,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.flatten()

    region_mask = np.full((h, w), -1, dtype=np.int32)
    region_mask[indices] = labels

    cluster_data = []
    for cluster_id in range(cluster_count):
        cluster_pixels = rgb[indices][labels == cluster_id]
        cluster_y = y_coords[labels == cluster_id]
        if cluster_pixels.size == 0:
            continue
        mean_color = tuple(int(c) for c in cluster_pixels.mean(axis=0)[::-1])
        cluster_data.append({
            "cluster_id": cluster_id,
            "mean_y": float(cluster_y.mean()) if cluster_y.size else 0.0,
            "mean_color": mean_color,
        })

    cluster_data.sort(key=lambda item: item["mean_y"], reverse=True)
    labels_by_order = ["outsole", "midsole", "upper", "detail"]
    label_by_cluster = {}
    region_colors: Dict[int, Tuple[int, int, int]] = {}
    region_name: Dict[int, str] = {}
    for rank, item in enumerate(cluster_data):
        region_id = rank
        region_name[region_id] = labels_by_order[min(rank, len(labels_by_order) - 1)]
        region_colors[region_id] = item["mean_color"]
        label_by_cluster[item["cluster_id"]] = region_id

    for idx in range(cluster_count):
        if idx not in label_by_cluster:
            label_by_cluster[idx] = 0

    region_mask[mask] = np.array([label_by_cluster[c] for c in labels], dtype=np.int32)

    print(f"[COLOR_EXTRACT] Tìm thấy {len(cluster_data)} vùng: {region_name}")
    print(f"[COLOR_EXTRACT] Màu trung bình: {region_colors}")
    return region_mask, region_colors, region_name


def _project_points_to_image_space(
    points: np.ndarray,
    camera_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    w = int(camera_params["image_size"][0])
    h = int(camera_params["image_size"][1])
    fx = float(camera_params["fx"])
    fy = float(camera_params["fy"])
    cx = float(camera_params["cx"])
    cy = float(camera_params["cy"])
    axis_map = tuple(camera_params["axis_map"])
    signs = tuple(camera_params["signs"])
    cam_dist = float(camera_params["cam_dist"])

    x_cam = points[:, axis_map[0]] * signs[0]
    y_cam = points[:, axis_map[1]] * signs[1]
    z_cam = points[:, axis_map[2]] * signs[2] - cam_dist
    valid = z_cam < 0
    u = cx + fx * (x_cam / -z_cam)
    v = cy + fy * (y_cam / -z_cam)
    projected = np.stack([u, v], axis=1)
    inside = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] <= w - 1)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] <= h - 1)
    )
    valid = valid & inside
    return projected, valid


def project_mesh_to_image_space(
    mesh: Any,
    camera_params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    mesh = _as_trimesh(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    return _project_points_to_image_space(vertices, camera_params)


def _triangle_region_from_centroid(
    projected_triangle: np.ndarray,
    region_mask: np.ndarray,
    default_region: int = 0,
) -> int:
    u = np.clip(int(round(projected_triangle[:, 0].mean())), 0, region_mask.shape[1] - 1)
    v = np.clip(int(round(projected_triangle[:, 1].mean())), 0, region_mask.shape[0] - 1)
    region_id = int(region_mask[v, u])
    return region_id if region_id >= 0 else default_region


def subdivide_and_colorize(
    mesh: Any,
    region_mask: np.ndarray,
    projected_coords: np.ndarray,
    region_colors: Dict[int, Tuple[int, int, int]],
    max_triangle_size: Optional[int] = None,
    max_depth: int = 2,
) -> Tuple[Any, np.ndarray]:
    mesh = _as_trimesh(mesh)
    h, w = region_mask.shape
    if max_triangle_size is None:
        max_triangle_size = max(16, min(w, h) // 24)

    vertices = [np.asarray(v, dtype=np.float32) for v in mesh.vertices]
    projected = [np.asarray(p, dtype=np.float32) for p in projected_coords]
    faces = [list(face) for face in mesh.faces]
    midpoint_cache = {}
    new_faces = []
    face_colors = []

    def _edge_length(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _get_midpoint(i0: int, i1: int) -> int:
        key = tuple(sorted((i0, i1)))
        if key in midpoint_cache:
            return midpoint_cache[key]
        v0, v1 = vertices[i0], vertices[i1]
        p0, p1 = projected[i0], projected[i1]
        vertices.append(((v0 + v1) * 0.5).astype(np.float32))
        projected.append(((p0 + p1) * 0.5).astype(np.float32))
        idx = len(vertices) - 1
        midpoint_cache[key] = idx
        return idx

    def _process_face(face_indices: Tuple[int, int, int], depth: int = 0) -> None:
        i0, i1, i2 = face_indices
        p0, p1, p2 = projected[i0], projected[i1], projected[i2]
        edges = [
            _edge_length(p0, p1),
            _edge_length(p1, p2),
            _edge_length(p2, p0),
        ]
        regions = [
            _sample_region_from_mask(p0, region_mask),
            _sample_region_from_mask(p1, region_mask),
            _sample_region_from_mask(p2, region_mask),
        ]
        if depth < max_depth and (
            max(edges) > max_triangle_size or len(set(regions)) > 1
        ):
            m01 = _get_midpoint(i0, i1)
            m12 = _get_midpoint(i1, i2)
            m20 = _get_midpoint(i2, i0)
            _process_face((i0, m01, m20), depth + 1)
            _process_face((m01, i1, m12), depth + 1)
            _process_face((m20, m12, i2), depth + 1)
            _process_face((m01, m12, m20), depth + 1)
            return

        region_id = _triangle_region_from_centroid(np.stack([p0, p1, p2], axis=0), region_mask)
        color = region_colors.get(region_id, (200, 200, 200))
        new_faces.append((i0, i1, i2))
        face_colors.append(color)

    for face in faces:
        _process_face(tuple(face), depth=0)

    final_vertices = np.asarray(vertices, dtype=np.float32)
    final_faces = np.asarray(new_faces, dtype=np.int64)
    mesh_out = trimesh.Trimesh(vertices=final_vertices, faces=final_faces, process=False)

    vertex_color_accum = [[] for _ in range(len(final_vertices))]
    for face_idx, face in enumerate(final_faces):
        color = np.array(face_colors[face_idx], dtype=np.float32)
        for vid in face:
            vertex_color_accum[vid].append(color)

    vertex_colors = np.zeros((len(final_vertices), 4), dtype=np.uint8)
    for idx, colors in enumerate(vertex_color_accum):
        if colors:
            averaged = np.clip(np.mean(colors, axis=0), 0, 255).astype(np.uint8)
            vertex_colors[idx, :3] = averaged
            vertex_colors[idx, 3] = 255
        else:
            vertex_colors[idx] = np.array([200, 200, 200, 255], dtype=np.uint8)

    return mesh_out, vertex_colors


def render_comparison(
    mesh: Any,
    camera_params: Dict[str, Any],
    reference_image_path: str,
    output_path: Optional[str] = None,
    show: bool = False,
) -> str:
    rgb, mask = _load_reference_image(reference_image_path)
    projected, valid = project_mesh_to_image_space(mesh, camera_params)
    mesh = _as_trimesh(mesh)
    vc = np.asarray(mesh.visual.vertex_colors)
    if vc.size == 0 or vc.shape[1] < 3:
        vc = np.tile(np.array([200, 200, 200, 255], dtype=np.uint8), (len(mesh.vertices), 1))

    overlay = rgb.copy()
    for idx, (u, v) in enumerate(projected):
        if not valid[idx]:
            continue
        x = int(round(u))
        y = int(round(v))
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            color = vc[idx, :3].astype(np.uint8)
            cv2.circle(overlay, (x, y), 1, tuple(int(c) for c in color[::-1]), -1)

    rows = [cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)]
    titles = ["Reference", "Projected mesh colors"]
    if output_path is None:
        out = Path(reference_image_path).with_suffix("_comparison.png")
    else:
        out = Path(output_path)
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, img, title in zip(axes, rows, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[COMPARE] Đã lưu hình so sánh: {out}")
    if show:
        try:
            import matplotlib.pyplot as plt
            img = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        except Exception:
            pass
    return str(out)


def load_mesh(input_path: str) -> Any:
    """Đọc mesh từ file đầu vào (.glb, .ply, .obj, ...)."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file mesh: {path}")

    if trimesh is None:
        raise ImportError("trimesh chưa được cài đặt")

    print(f"[MESH] Đang đọc mesh từ: {path}")
    mesh = trimesh.load_mesh(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise ValueError("Scene mesh không có geometry hợp lệ")
        mesh = trimesh.util.concatenate(geometries)
    return mesh


def _simple_kmeans(points: np.ndarray, n_clusters: int = 3, max_iters: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """KMeans đơn giản trên numpy để gom các pixel màu và vị trí theo chiều cao thành các nhóm."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, points.shape[1]), dtype=np.float32)
    if len(points) < n_clusters:
        n_clusters = max(1, len(points))

    rng = np.random.default_rng(0)
    centers = points[rng.choice(len(points), size=n_clusters, replace=False)].copy()

    for _ in range(max_iters):
        distances = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.empty_like(centers)
        for cluster_idx in range(n_clusters):
            members = points[labels == cluster_idx]
            if len(members) == 0:
                new_centers[cluster_idx] = centers[cluster_idx]
            else:
                new_centers[cluster_idx] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels, centers


def _extract_reference_colors_from_image(image_path: str, cluster_count: int = 3) -> Tuple[list[Dict[str, Any]], list[float]]:
    """Trích màu đại diện từ ảnh gốc theo chiều cao bằng clustering màu.
    
    Chiều cao ảnh được chuẩn hóa từ 0 (đáy/dưới) đến 1 (trên/cổ giày).
    Boundaries sẽ chia 3 vùng: outsole (0-30%), midsole (30-50%), upper (50-100%).
    """
    if not image_path:
        return [], []
    path = Path(image_path)
    if not path.exists():
        return [], []

    image = Image.open(str(path)).convert("RGBA")
    image_np = np.array(image, dtype=np.uint8)
    if image_np.ndim != 3 or image_np.shape[2] < 4:
        return [], []

    alpha = image_np[:, :, 3] > 0
    if not np.any(alpha):
        alpha = np.ones_like(image_np[:, :, 3], dtype=bool)

    # Nếu ảnh không có alpha channel, loại bỏ nền tự động bằng màu viền.
    if alpha.all():
        border_pixels = np.concatenate(
            [image_np[0, :, :3], image_np[-1, :, :3], image_np[:, 0, :3], image_np[:, -1, :3]],
            axis=0,
        )
        bg_color = np.median(border_pixels.astype(np.float32), axis=0)
        dist = np.linalg.norm(image_np[:, :, :3].astype(np.float32) - bg_color[None, None, :], axis=2)
        alpha = dist > 20.0
        if not np.any(alpha):
            alpha = np.ones_like(alpha, dtype=bool)

    pixels = image_np[alpha, :3].astype(np.float32) / 255.0
    ys = np.where(alpha)[0]
    if len(pixels) == 0:
        return [], []

    # Từ đế lên cổ giày: y pixel càng lớn thì ở phía dưới ảnh, nên chuẩn hóa NGƯỢC LẠI
    # để y_ratio = 0 là đáy và y_ratio = 1 là trên cùng.
    y_min = float(ys.min())  # Trên cùng ảnh (giá trị pixel nhỏ)
    y_max = float(ys.max())  # Dưới cùng ảnh (giá trị pixel lớn)
    if np.isclose(y_max, y_min):
        y_ratio = np.zeros(len(pixels), dtype=np.float32)
    else:
        # y_pixel nhỏ → cao trên ảnh → y_ratio lớn (1.0)
        # y_pixel lớn → thấp dưới ảnh → y_ratio nhỏ (0.0)
        y_ratio = (y_max - ys.astype(np.float32)) / (y_max - y_min)

    features = np.column_stack([pixels, y_ratio[:, None]])
    labels, centers = _simple_kmeans(features, n_clusters=min(cluster_count, len(features)))

    cluster_data = []
    for cluster_idx in range(centers.shape[0]):
        members = features[labels == cluster_idx]
        if len(members) == 0:
            continue
        color = np.clip(members[:, :3].mean(axis=0), 0.0, 1.0)
        height = float(members[:, 3].mean())
        cluster_data.append({"color": color, "height": height})

    if not cluster_data:
        # Fallback: chia 3 vùng bằng nhau
        return [
            {"label": "outsole", "color": np.array([0.2, 0.2, 0.2])},
            {"label": "midsole", "color": np.array([0.8, 0.8, 0.8])},
            {"label": "upper", "color": np.array([0.4, 0.4, 0.8])},
        ], [0.0, 0.3, 0.5, 1.0]

    cluster_data.sort(key=lambda item: item["height"])
    colors = []
    if len(cluster_data) == 2:
        label_order = ["outsole", "upper"]
    else:
        label_order = ["outsole", "midsole", "upper"]

    for cluster_idx, item in enumerate(cluster_data):
        label = label_order[min(cluster_idx, len(label_order) - 1)]
        colors.append({
            "label": label,
            "color": item["color"],
            "height": item["height"],
        })

    # Chia 3 vùng dựa trên height từ clustering
    if len(cluster_data) >= 2:
        boundaries = [0.0]
        for idx in range(len(cluster_data) - 1):
            mid = (cluster_data[idx]["height"] + cluster_data[idx + 1]["height"]) / 2.0
            boundaries.append(float(np.clip(mid, 0.0, 1.0)))
        boundaries.append(1.0)
    else:
        # Fallback: chia đều 3 vùng
        boundaries = [0.0, 0.3, 0.5, 1.0]

    print(f"[COLOR_EXTRACT] boundaries={boundaries} cluster_heights={[item['height'] for item in cluster_data]}")
    return colors, boundaries


def _otsu_threshold(values: np.ndarray) -> float:
    """Tìm ngưỡng phân tách 1D bằng thuật toán Otsu trên phân phối giá trị hình học."""
    values = np.asarray(values, dtype=np.float32).ravel()
    if values.size < 2:
        return float(values.mean()) if values.size else 0.0

    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(min_value, max_value):
        return min_value

    bins = max(2, min(20, values.size))
    hist, bin_edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total <= 0:
        return float(np.median(values))

    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(len(hist)))
    total_mean = cumulative_mean[-1]

    best_threshold = 0.0
    best_between = -1.0
    for idx in range(len(hist) - 1):
        weight_bg = cumulative_sum[idx]
        weight_fg = total - weight_bg
        if weight_bg <= 0 or weight_fg <= 0:
            continue
        mean_bg = cumulative_mean[idx] / weight_bg
        mean_fg = (total_mean - cumulative_mean[idx]) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_between:
            best_between = between
            best_threshold = float((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)

    return best_threshold


def compute_bbox(mesh: Any, axis: str = "auto") -> Dict[str, Any]:
    """Tính bounding box và chọn trục chiều cao phù hợp bằng phân tích hình học 3D."""
    mesh = _as_trimesh(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.size == 0:
        raise ValueError("Mesh không có vertex")

    bbox = {
        "x_min": float(vertices[:, 0].min()),
        "x_max": float(vertices[:, 0].max()),
        "y_min": float(vertices[:, 1].min()),
        "y_max": float(vertices[:, 1].max()),
        "z_min": float(vertices[:, 2].min()),
        "z_max": float(vertices[:, 2].max()),
    }
    bbox["ranges"] = {
        "x": bbox["x_max"] - bbox["x_min"],
        "y": bbox["y_max"] - bbox["y_min"],
        "z": bbox["z_max"] - bbox["z_min"],
    }

    if axis == "auto":
        # Thuật toán mới không dựa vào index/UV. Chúng ta chọn trục có phân bố chiều cao
        # rõ ràng nhất bằng cách so sánh độ tách lớp giữa phần thấp và phần cao của mesh.
        candidates = []
        for axis_name in ("x", "y", "z"):
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]
            coords = vertices[:, axis_idx]
            height_min = float(coords.min())
            height_max = float(coords.max())
            height_range = height_max - height_min
            if height_range <= 1e-6:
                continue
            split_value = _otsu_threshold(coords)
            relative = (coords - height_min) / height_range
            split_relative = (split_value - height_min) / height_range
            lower_mask = relative <= split_relative
            upper_mask = ~lower_mask
            if lower_mask.sum() < 0.05 * len(vertices) or upper_mask.sum() < 0.05 * len(vertices):
                score = 0.0
            else:
                lower_mean = float(relative[lower_mask].mean())
                upper_mean = float(relative[upper_mask].mean())
                score = (lower_mask.sum() / len(vertices)) * (upper_mask.sum() / len(vertices)) * (upper_mean - lower_mean) ** 2
            candidates.append((score, axis_name, split_value, height_range))

        if candidates:
            best_score, best_axis, best_split, _ = max(candidates, key=lambda item: item[0])
            height_axis = best_axis
            split_value = best_split
        else:
            height_axis = "y"
            split_value = 0.0
    elif axis in {"x", "y", "z"}:
        height_axis = axis
        split_value = 0.0
    else:
        raise ValueError("axis phải là 'auto', 'x', 'y' hoặc 'z'")

    bbox["height_axis"] = height_axis
    bbox["height_min"] = bbox[f"{height_axis}_min"]
    bbox["height_max"] = bbox[f"{height_axis}_max"]
    bbox["height_range"] = bbox["ranges"][height_axis]
    bbox["auto_split_value"] = float(split_value)
    return bbox


def segment_by_height(
    mesh: Any,
    threshold: float = 0.15,
    threshold_mode: str = "ratio",
    axis: str = "auto",
    auto_bins: int = 20,
    image_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Phân đoạn vertex thành outsole/midsole/upper bằng hình học không gian.

    Thuật toán mới không dùng UV, face index hay vertex order. Nó dựa vào hai tín hiệu:
    1) chiều cao 3D của từng vertex trong mesh,
    2) màu/độ cao của ảnh gốc để biết vùng nào ở đáy, giữa, đỉnh của chiếc giày.
    Điều này cho phép ánh xạ màu thực tế từ ảnh gốc vào mesh trắng mà không bị lệch.
    """
    mesh = _as_trimesh(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    bbox = compute_bbox(mesh, axis=axis)

    axis_name = bbox["height_axis"]
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]
    coords = vertices[:, axis_idx]
    height_min = float(coords.min())
    height_max = float(coords.max())
    height_range = height_max - height_min

    if height_range <= 1e-6:
        labels = np.full(len(vertices), "outsole", dtype=object)
        bbox["split_value"] = float(height_min)
        bbox["threshold_mode"] = threshold_mode
        bbox["threshold"] = float(threshold)
        return labels, bbox

    relative_height = (coords - height_min) / height_range

    if image_path:
        reference_colors, boundaries = _extract_reference_colors_from_image(image_path)
        if len(reference_colors) >= 2 and len(boundaries) >= 3:
            labels = np.empty(len(vertices), dtype=object)
            if len(reference_colors) == 2:
                for idx, rel in enumerate(relative_height):
                    labels[idx] = "outsole" if rel <= boundaries[1] else "upper"
            else:
                for idx, rel in enumerate(relative_height):
                    if rel <= boundaries[1]:
                        labels[idx] = "outsole"
                    elif rel <= boundaries[2]:
                        labels[idx] = "midsole"
                    else:
                        labels[idx] = "upper"
            bbox["region_boundaries"] = boundaries
            bbox["reference_colors"] = reference_colors
            bbox["split_value"] = float(boundaries[1] if len(boundaries) > 1 else 0.3)
            bbox["threshold_mode"] = "image"
            bbox["threshold"] = float(threshold)
            print(f"[SEGMENT] Using image-based boundaries: {boundaries} labels={len(reference_colors)}")
            return labels, bbox

    if threshold_mode == "ratio":
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold_mode='ratio' yêu cầu threshold trong [0, 1]")
        split_value = height_min + threshold * height_range
    elif threshold_mode == "absolute":
        split_value = float(threshold)
    elif threshold_mode == "auto":
        split_value = _otsu_threshold(coords)
        if np.isclose(split_value, height_min) or np.isclose(split_value, height_max):
            split_value = height_min + 0.18 * height_range
    else:
        raise ValueError("threshold_mode phải là 'ratio', 'absolute' hoặc 'auto'")

    split_relative = (split_value - height_min) / height_range
    labels = np.where(relative_height <= split_relative, "outsole", "upper")
    bbox["split_value"] = float(split_value)
    bbox["split_relative"] = float(split_relative)
    bbox["threshold_mode"] = threshold_mode
    bbox["threshold"] = float(threshold)
    return labels, bbox


def assign_colors(
    mesh: Any,
    labels: Optional[np.ndarray] = None,
    sole_color: Any = (60, 60, 60, 255),
    upper_color: Any = (200, 30, 30, 255),
    threshold: float = 0.15,
    threshold_mode: str = "ratio",
    axis: str = "auto",
    auto_bins: int = 20,
    image_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Gán màu RGBA cho từng vertex theo vùng outsole/midsole/upper, ưu tiên màu từ ảnh gốc."""
    mesh = _as_trimesh(mesh)
    if labels is None:
        labels, bbox = segment_by_height(
            mesh,
            threshold=threshold,
            threshold_mode=threshold_mode,
            axis=axis,
            auto_bins=auto_bins,
            image_path=image_path,
        )
    else:
        bbox = compute_bbox(mesh, axis=axis)

    colors = np.empty((len(labels), 4), dtype=np.uint8)

    if image_path:
        reference_colors, _ = _extract_reference_colors_from_image(image_path)
        if len(reference_colors) >= 3:
            color_map = {}
            for item in reference_colors:
                label_name = item["label"]
                color_map[label_name] = np.clip(np.array(item["color"]), 0.0, 1.0)
            for idx, label in enumerate(labels):
                if label in color_map:
                    rgba = np.round(np.array(color_map[label]) * 255.0).astype(np.uint8)
                    colors[idx] = np.array([rgba[0], rgba[1], rgba[2], 255], dtype=np.uint8)
                else:
                    colors[idx] = np.array([255, 255, 255, 255], dtype=np.uint8)
            return colors, labels, bbox

    sole_rgba = _normalize_rgba(sole_color)
    upper_rgba = _normalize_rgba(upper_color)
    colors[labels == "outsole"] = sole_rgba
    colors[labels == "midsole"] = sole_rgba
    colors[labels == "upper"] = upper_rgba
    return colors, labels, bbox


def export_colored_mesh(
    mesh: Any,
    output_path: str,
    vertex_colors: Optional[np.ndarray] = None,
    file_type: Optional[str] = None,
) -> str:
    """Xuất mesh đã có vertex_colors sang file .ply hoặc .obj/.mtl."""
    mesh = _as_trimesh(mesh)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors

    if file_type is None:
        suffix = out_path.suffix.lower()
        file_type = "glb" if suffix == ".glb" else "obj" if suffix == ".obj" else "ply" if suffix == ".ply" else "glb"

    if file_type == "ply":
        mesh.export(str(out_path), file_type="ply")
    elif file_type == "obj":
        mesh.export(str(out_path), file_type="obj")
    elif file_type == "glb":
        mesh.export(str(out_path), file_type="glb")
    else:
        raise ValueError("file_type phải là 'ply', 'obj' hoặc 'glb'")

    print(f"[EXPORT] Đã xuất mesh màu: {out_path}")
    return str(out_path)


def visualize(mesh: Any, vertex_colors: Optional[np.ndarray] = None, show: bool = True) -> None:
    """Hiển thị nhanh mesh/vertex colors bằng open3d nếu có, fallback sang matplotlib."""
    mesh = _as_trimesh(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertex_colors is None:
        vertex_colors = np.tile(np.array([128, 128, 128, 255], dtype=np.uint8), (len(vertices), 1))

    try:
        import open3d as o3d

        tri_mesh = o3d.geometry.TriangleMesh()
        tri_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        tri_mesh.triangles = o3d.utility.Vector3iVector(faces)
        tri_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors[:, :3] / 255.0)
        if show:
            o3d.visualization.draw_geometries([tri_mesh])
        return
    except Exception:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=vertex_colors[:, :3] / 255.0, s=8)
            ax.set_title("Mesh vertex colors")
            plt.show()
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"[VIS] Không thể hiển thị trực quan: {exc}")


def apply_uv_mapping(
    mesh: Any,
    model: Optional[Any] = None,
    scene_code: Optional[Any] = None,
    out_dir: Optional[str] = None,
    stem: Optional[str] = None,
    texture_res: int = 2048,
    original_image_path: Optional[str] = None,
    threshold: float = 0.15,
    threshold_mode: str = "ratio",
    axis: str = "auto",
    sole_color: Any = (60, 60, 60, 255),
    upper_color: Any = (200, 30, 30, 255),
    export_path: Optional[str] = None,
    visualize_result: bool = False,
) -> str:
    """Wrapper tương thích với pipeline hiện tại: tô màu mesh bằng ảnh tham chiếu hoặc theo chiều cao."""
    mesh = _as_trimesh(mesh)
    if out_dir is None:
        out_dir = "."
    if stem is None:
        stem = "colored_mesh"

    output_path = export_path or str(Path(out_dir) / f"{stem}_colored.glb")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if original_image_path is not None and Path(original_image_path).exists():
        try:
            camera_params = align_camera_to_reference(mesh, original_image_path)
            region_mask, region_colors, region_names = extract_color_regions(original_image_path, cluster_count=3)
            projected_coords, valid = project_mesh_to_image_space(mesh, camera_params)
            colored_mesh, vertex_colors = subdivide_and_colorize(
                mesh,
                region_mask,
                projected_coords,
                region_colors,
                max_triangle_size=min(48, max(region_mask.shape) // 16),
                max_depth=2,
            )
            export_colored_mesh(colored_mesh, output_path, vertex_colors=vertex_colors)
            print(f"[UV_MAPPING] Áp dụng màu từ ảnh tham chiếu '{original_image_path}' với {len(region_colors)} vùng")
            comparison_path = str(Path(out_dir) / f"{stem}_comparison.png")
            render_comparison(colored_mesh, camera_params, original_image_path, output_path=comparison_path, show=False)
            if visualize_result:
                visualize(colored_mesh, vertex_colors=vertex_colors, show=True)
            return output_path
        except Exception as exc:
            print(f"[UV_MAPPING] Không thể áp dụng colorization theo ảnh tham chiếu: {exc}")
            print("[UV_MAPPING] Chuyển sang phân đoạn chiều cao làm fallback...")

    labels, bbox = segment_by_height(
        mesh,
        threshold=threshold,
        threshold_mode=threshold_mode,
        axis=axis,
        image_path=original_image_path,
    )
    colors, labels, bbox = assign_colors(
        mesh,
        labels=labels,
        sole_color=sole_color,
        upper_color=upper_color,
        threshold=threshold,
        threshold_mode=threshold_mode,
        axis=axis,
        image_path=original_image_path,
    )

    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = colors
    export_colored_mesh(colored_mesh, output_path, vertex_colors=colors)

    if visualize_result:
        visualize(colored_mesh, vertex_colors=colors, show=True)

    print(f"[SEGMENT] axis={bbox['height_axis']} split={bbox.get('split_value', 0.0):.4f} labels={int(np.sum(labels == 'outsole'))} outsole / {int(np.sum(labels == 'midsole'))} midsole / {int(np.sum(labels == 'upper'))} upper")
    return output_path


def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Đọc tham số từ file JSON nếu có."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("File config phải là JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Tô màu mesh giày theo đế và thân")
    parser.add_argument("input_mesh", help="Đường dẫn mesh đầu vào")
    parser.add_argument("-o", "--output", default=None, help="Đường dẫn mesh đầu ra (.ply hoặc .obj)")
    parser.add_argument("--threshold", type=float, default=0.15, help="Ngưỡng chia đế/thân, ví dụ 0.15 = 15%% chiều cao")
    parser.add_argument("--threshold-mode", default="ratio", choices=["ratio", "absolute", "auto"], help="Cách xác định ngưỡng")
    parser.add_argument("--axis", default="auto", choices=["auto", "x", "y", "z"], help="Trục chiều cao dùng để phân đoạn")
    parser.add_argument("--original-image", default=None, help="Đường dẫn ảnh gốc hoặc ảnh đã tách nền để dùng phân đoạn và tô màu")
    parser.add_argument("--sole-color", nargs=4, type=int, default=[60, 60, 60, 255], help="Màu đế dạng RGBA")
    parser.add_argument("--upper-color", nargs=4, type=int, default=[200, 30, 30, 255], help="Màu thân dạng RGBA")
    parser.add_argument("--config", default=None, help="Đường dẫn file JSON chứa tham số")
    parser.add_argument("--visualize", action="store_true", help="Hiển thị kết quả sau khi xuất")
    args = parser.parse_args()

    config = _load_config(args.config)
    threshold = config.get("threshold", args.threshold)
    threshold_mode = config.get("threshold_mode", args.threshold_mode)
    axis = config.get("axis", args.axis)
    original_image_path = config.get("original_image", args.original_image)
    sole_color = tuple(config.get("sole_color", tuple(args.sole_color)))
    upper_color = tuple(config.get("upper_color", tuple(args.upper_color)))
    output_path = config.get("output", args.output)

    mesh = load_mesh(args.input_mesh)
    labels, bbox = segment_by_height(
        mesh,
        threshold=float(threshold),
        threshold_mode=str(threshold_mode),
        axis=str(axis),
        image_path=original_image_path,
    )
    colors, _, _ = assign_colors(
        mesh,
        labels=labels,
        sole_color=sole_color,
        upper_color=upper_color,
        threshold=float(threshold),
        threshold_mode=str(threshold_mode),
        axis=str(axis),
        image_path=original_image_path,
    )

    if output_path is None:
        output_path = str(Path(args.input_mesh).with_suffix(".glb"))
    export_colored_mesh(mesh, output_path, vertex_colors=colors)
    print(f"[DONE] Bbox: {bbox}")
    if args.visualize or config.get("visualize", False):
        visualize(mesh, vertex_colors=colors, show=True)


if __name__ == "__main__":
    main()
