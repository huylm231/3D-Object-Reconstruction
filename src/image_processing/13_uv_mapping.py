import numpy as np
from pathlib import Path
from PIL import Image


def _fallback_vertex_colors(mesh, model, scene_code, out_dir, stem, trimesh):
    """Fallback: tô màu bằng Vertex Colors khi bake_texture thất bại."""
    meshes = model.extract_mesh(scene_code.unsqueeze(0), True, resolution=256)
    fallback_mesh = meshes[0]
    
    vertex_colors = np.array(fallback_mesh.visual.vertex_colors)
    if vertex_colors.shape[1] == 4:
        # QUAN TRỌNG: Ép toàn bộ alpha = 255 để mesh ĐỤC HOÀN TOÀN,
        # không cho nhìn xuyên vào bên trong mô hình.
        vertex_colors[:, 3] = 255
        
    colored_trimesh = trimesh.Trimesh(
        vertices=np.array(fallback_mesh.vertices),
        faces=np.array(fallback_mesh.faces),
        vertex_colors=vertex_colors,
        process=False,
    )
    colored_trimesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    colored_trimesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    
    glb_path = Path(out_dir) / f"{stem}_textured.glb"
    colored_trimesh.export(str(glb_path), file_type='glb')
    if not glb_path.exists():
        print(f"[UV MAPPING] warning: textured GLB export failed, file missing: {glb_path}")
    else:
        print(f"[UV MAPPING] exported textured GLB: {glb_path}")
    return str(glb_path)


def _find_best_orthographic_mapping(vertices, normals, image_alpha, normal_threshold: float = 0.05):
    """Find the best axis mapping for orthographic vertex projection into the input image."""
    from itertools import permutations, product

    best_score = -1.0
    best_mapping = None
    best_normals = None

    img_h, img_w = image_alpha.shape
    for ax_u, ax_v, ax_d in permutations((0, 1, 2), 3):
        for sign_u, sign_v, sign_d in product((1, -1), repeat=3):
            coords_u = vertices[:, ax_u] * sign_u
            coords_v = vertices[:, ax_v] * sign_v

            span_u = coords_u.max() - coords_u.min()
            span_v = coords_v.max() - coords_v.min()
            if span_u < 1e-6 or span_v < 1e-6:
                continue

            u_norm = (coords_u - coords_u.min()) / span_u
            v_norm = (coords_v - coords_v.min()) / span_v
            u_px = np.clip(np.rint(u_norm * (img_w - 1)).astype(np.int32), 0, img_w - 1)
            v_px = np.clip(np.rint((1.0 - v_norm) * (img_h - 1)).astype(np.int32), 0, img_h - 1)
            valid = image_alpha[v_px, u_px]
            coverage = int(np.count_nonzero(valid))
            if coverage == 0:
                continue

            score = float(coverage)
            if normals is not None:
                front = normals[:, ax_d] * sign_d < -normal_threshold
                score += 0.01 * np.count_nonzero(valid & front)

            if score > best_score:
                best_score = score
                best_mapping = (ax_u, ax_v, ax_d, sign_u, sign_v, sign_d)
                best_normals = normals[:, ax_d] if normals is not None else None

    return best_mapping


def _project_vertex_colors(mesh, image_path, out_dir, stem, normal_threshold: float = 0.05):
    """Project mesh vertices orthographically to the input image and assign direct pixel colors."""
    import trimesh
    from itertools import permutations, product

    image = Image.open(image_path).convert("RGBA")
    image_np = np.array(image)
    H, W = image_np.shape[:2]
    image_alpha = image_np[:, :, 3] > 0

    vertices = np.array(mesh.vertices)
    if vertices.size == 0:
        raise ValueError("Mesh has no vertices for direct pixel projection.")

    if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) == len(vertices):
        normals = np.array(mesh.vertex_normals)
    else:
        mesh_copy = mesh.copy()
        try:
            mesh_copy.fix_normals()
        except Exception:
            pass
        normals = np.array(mesh_copy.vertex_normals) if len(mesh_copy.vertex_normals) == len(vertices) else None

    def _fit_to_image(coords_u, coords_v):
        min_u, max_u = coords_u.min(), coords_u.max()
        min_v, max_v = coords_v.min(), coords_v.max()
        span_u = max(max_u - min_u, 1e-6)
        span_v = max(max_v - min_v, 1e-6)
        obj_aspect = span_u / span_v
        img_aspect = W / H

        u_norm = (coords_u - min_u) / span_u
        v_norm = (coords_v - min_v) / span_v
        if obj_aspect > img_aspect:
            v_norm = 0.5 + (v_norm - 0.5) * (img_aspect / obj_aspect)
        else:
            u_norm = 0.5 + (u_norm - 0.5) * (obj_aspect / img_aspect)
        return u_norm, v_norm

    best_score = -1.0
    best_mapping = None
    best_uv = None
    for ax_u, ax_v, ax_d in permutations((0, 1, 2), 3):
        for sign_u, sign_v, sign_d in product((1, -1), repeat=3):
            coords_u = vertices[:, ax_u] * sign_u
            coords_v = vertices[:, ax_v] * sign_v
            u_norm, v_norm = _fit_to_image(coords_u, coords_v)
            u_px = np.clip(np.rint(u_norm * (W - 1)).astype(np.int32), 0, W - 1)
            v_px = np.clip(np.rint((1.0 - v_norm) * (H - 1)).astype(np.int32), 0, H - 1)
            valid = image_alpha[v_px, u_px]
            coverage = int(np.count_nonzero(valid))
            if coverage == 0:
                continue
            score = float(coverage)
            if normals is not None:
                front = normals[:, ax_d] * sign_d < -normal_threshold
                score += 0.1 * np.count_nonzero(valid & front)
            if score > best_score:
                best_score = score
                best_mapping = (ax_u, ax_v, ax_d, sign_u, sign_v, sign_d)
                best_uv = (u_px, v_px)

    if best_mapping is None or best_uv is None:
        raise RuntimeError("Could not determine a robust orthographic projection mapping for direct pixel sampling.")

    ax_u, ax_v, ax_d, sign_u, sign_v, sign_d = best_mapping
    u, v = best_uv
    sampled_colors = image_np[v, u].copy()
    sampled_colors[:, 3] = 255

    if hasattr(mesh.visual, 'vertex_colors') and len(mesh.visual.vertex_colors) == len(vertices):
        base_colors = np.array(mesh.visual.vertex_colors, dtype=np.uint8)
    else:
        base_colors = np.tile(np.array([128, 128, 128, 255], dtype=np.uint8), (len(vertices), 1))

    visible = image_alpha[v, u]
    if normals is not None:
        front = normals[:, ax_d] * sign_d < -normal_threshold
        visible = visible & front

    if np.any(visible):
        base_colors[visible] = sampled_colors[visible]

    print(f"[UV MAPPING] direct sampling mapping axes=({ax_u},{ax_v},{ax_d}) signs=({sign_u},{sign_v},{sign_d}) score={best_score:.1f} visible={int(np.sum(visible))}/{len(vertices)}")

    colored_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=np.array(mesh.faces),
        vertex_colors=base_colors,
        process=False,
    )
    colored_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    colored_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))

    glb_path = Path(out_dir) / f"{stem}_pixel_colored.glb"
    colored_mesh.export(str(glb_path), file_type='glb')
    if not glb_path.exists():
        print(f"[UV MAPPING] warning: direct pixel-colored GLB export failed: {glb_path}")
    else:
        print(f"[UV MAPPING] exported direct pixel-colored GLB: {glb_path}")
    return str(glb_path)


def apply_uv_mapping(mesh, model, scene_code, out_dir, stem, texture_res=2048, original_image_path=None):
    """
    Áp dụng thuật toán LSCM UV Unwrapping (qua xatlas) và Bake màu thực tế của AI lên lưới 3D.
    Nếu thiếu thư viện hoặc lỗi device CUDA/CPU, tự động fallback sang Vertex Colors.
    """
    # pyrefly: ignore [missing-import]
    import trimesh
    import sys
    
    try:
        import xatlas
        sys.path.append(str(Path(__file__).parent.parent))
        from tsr.bake_texture import bake_texture

        # If original image is provided, generate per-region masks automatically so bake can use them.
        if original_image_path is not None:
            try:
                import importlib.util
                seg_path = Path(__file__).parent / "14_region_segmentation.py"
                if seg_path.exists():
                    spec = importlib.util.spec_from_file_location("region_segmentation", seg_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    segment_shoe_three_regions = module.segment_shoe_three_regions

                    original_path = Path(original_image_path)
                    mask_dir = original_path.parent
                    stem_orig = original_path.stem
                    expected_masks = [mask_dir / f"{stem_orig}_mask_{r}.png" for r in ("sole", "body", "collar")]
                    if not all(p.exists() for p in expected_masks):
                        print(f"[UV MAPPING] generating region masks for {original_image_path}")
                        segment_shoe_three_regions(original_image_path, out_dir=mask_dir)
                else:
                    print(f"[UV MAPPING] region segmentation script not found: {seg_path}")
            except Exception as e:
                print(f"[UV MAPPING] region mask generation failed: {e}")

        # Tạo lưới atlas 2D và render màu
        bake_result = bake_texture(mesh, model, scene_code, texture_res, original_image_path=original_image_path)

    except (ImportError, RuntimeError) as e:
        print(f"  [CẢNH BÁO] Bake texture thất bại ({type(e).__name__}: {e}).")
        if original_image_path is not None:
            try:
                print("  [UV MAPPING] Chuyển sang direct pixel sampling vertex colors từ ảnh gốc...")
                return _project_vertex_colors(mesh, original_image_path, out_dir, stem)
            except Exception as e2:
                print(f"  [UV MAPPING] direct pixel sampling thất bại: {e2}")
        print("  [CẢNH BÁO] Chuyển sang dùng màu Vertex Colors để không làm gián đoạn tiến trình!")
        return _fallback_vertex_colors(mesh, model, scene_code, out_dir, stem, trimesh)

    if original_image_path is not None:
        try:
            print("  [UV MAPPING] Áp dụng direct pixel sampling vertex colors ngay cả khi bake texture thành công...")
            return _project_vertex_colors(mesh, original_image_path, out_dir, stem)
        except Exception as e2:
            print(f"  [UV MAPPING] direct pixel sampling fallback failed: {e2}")
    
    # --- Nhánh bake texture qua xatlas (nếu thành công) ---
    vmapping = bake_result["vmapping"]
    indices = bake_result["indices"]
    uvs = bake_result["uvs"].copy()
    colors = bake_result["colors"]  # Float32 [0, 1] shape (H, W, 4)
    
    # Khởi tạo mesh mới dựa trên chỉ mục đã unwrap của xatlas
    new_vertices = mesh.vertices[vmapping]
    new_faces = indices
    clean_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False,
    )
    # Căn chỉnh hệ tọa độ TripoSR sang chuẩn GLB (Y-up)
    clean_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    clean_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    # KHÔNG dùng fix_normals() vì thuật toán tự động này đoán sai chiều của lưới 3D hở
    # gây ra lỗi tàng hình / nhìn xuyên thấu (Backface culling)!
    
    # Chuyển đổi mảng màu Float32 sang ảnh PIL để làm Texture
    colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Nếu có kênh alpha, ép alpha về 255 để texture không bị trong suốt do vùng không dùng.
    if colors_uint8.shape[2] == 4:
        colors_uint8[:, :, 3] = 255
        texture_image = Image.fromarray(colors_uint8, mode="RGBA")
    else:
        texture_image = Image.fromarray(colors_uint8, mode="RGB")

    # Lật texture để đưa ảnh về đúng chiều, và lật UV để khớp với chuẩn GLB.
    texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
    uvs = uvs.copy()
    uvs[:, 1] = 1.0 - uvs[:, 1]

    # Lưu texture debug để kiểm tra chất lượng
    texture_image.save(str(Path(out_dir) / f"{stem}_texture_debug.png"))
    
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_image,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode='BLEND' if colors_uint8.shape[2] == 4 else 'OPAQUE',
        doubleSided=True
    )
    clean_mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs, 
        image=texture_image, 
        material=material
    )
    
    glb_path = Path(out_dir) / f"{stem}_textured.glb"
    clean_mesh.export(str(glb_path), file_type='glb')
    
    return str(glb_path)
