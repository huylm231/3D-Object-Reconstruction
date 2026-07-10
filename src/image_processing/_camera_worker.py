import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


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
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int32)
    else:
        indices = np.arange(len(faces), dtype=np.int32)

    polygons = []
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
        polygons.append(pts_i)

    if polygons:
        cv2.fillPoly(silhouette, polygons, 255)
    return silhouette > 0


def _render_projected_silhouette_from_faces(
    projected: np.ndarray,
    valid: np.ndarray,
    faces: np.ndarray,
    image_shape: Tuple[int, int],
    max_faces: int = 8000,
) -> np.ndarray:
    h, w = image_shape
    silhouette = np.zeros((h, w), dtype=np.uint8)
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int32)
    else:
        indices = np.arange(len(faces), dtype=np.int32)

    polygons = []
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
        polygons.append(pts_i)

    if polygons:
        cv2.fillPoly(silhouette, polygons, 255)
    return silhouette > 0


def _evaluate_camera_candidate(
    vertices: np.ndarray,
    faces: np.ndarray,
    silhouette_ref: np.ndarray,
    w: int,
    h: int,
    axis_map: Tuple[int, int, int],
    sign: Tuple[int, int, int],
    cam_dist: float,
    fovy_deg: float,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    tan_half = math.tan(math.radians(fovy_deg) / 2.0)
    if tan_half <= 0:
        return -1.0, None
    fx = 0.5 * w / tan_half
    fy = 0.5 * h / tan_half
    x_cam = vertices[:, axis_map[0]] * sign[0]
    y_cam = vertices[:, axis_map[1]] * sign[1]
    z_cam = vertices[:, axis_map[2]] * sign[2] - cam_dist
    valid = z_cam < 0
    if np.sum(valid) < max(10, len(vertices) * 0.05):
        return -1.0, None
    u = (w * 0.5) + fx * (x_cam / -z_cam)
    v = (h * 0.5) + fy * (y_cam / -z_cam)
    projected = np.stack([u, v], axis=1)
    silhouette_pred = _render_projected_silhouette_from_faces(
        projected, valid, faces, (h, w)
    )
    score = _silhouette_iou(silhouette_ref, silhouette_pred)
    params = {
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
    return score, params
