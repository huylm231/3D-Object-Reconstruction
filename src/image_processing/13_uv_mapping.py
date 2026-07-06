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
    
    glb_path = Path(out_dir) / f"{stem}_textured.glb"
    colored_trimesh.export(str(glb_path), file_type='glb')
    return str(glb_path)


def apply_uv_mapping(mesh, model, scene_code, out_dir, stem, texture_res=1024):
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
        
        # Tạo lưới atlas 2D và render màu
        bake_result = bake_texture(mesh, model, scene_code, texture_res)

    except (ImportError, RuntimeError) as e:
        print(f"  [CẢNH BÁO] Bake texture thất bại ({type(e).__name__}: {e}).")
        print("  [CẢNH BÁO] Chuyển sang dùng màu Vertex Colors để không làm gián đoạn tiến trình!")
        return _fallback_vertex_colors(mesh, model, scene_code, out_dir, stem, trimesh)
    
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
    # KHÔNG dùng fix_normals() vì thuật toán tự động này đoán sai chiều của lưới 3D hở
    # gây ra lỗi tàng hình / nhìn xuyên thấu (Backface culling)!
    
    # Chuyển đổi mảng màu Float32 sang ảnh PIL để làm Texture
    colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Giữ nguyên kênh Alpha của texture bake (nền trong suốt) để khi thu phóng (mipmap)
    # không bị lem màu đen từ viền vào trong UV islands (UV bleeding).

    # QUAN TRỌNG: Cần lật CẢ UV (đã lật ở trên) VÀ TEXTURE!
    # Vì thuật toán Moderngl trả về mảng màu bị lộn ngược so với chuẩn của ảnh.
    # Lật texture để đưa ảnh về đúng chiều, và lật UV để khớp với chuẩn GLB.
    texture_image = Image.fromarray(colors_uint8, mode="RGBA")
    texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Lưu texture debug để kiểm tra chất lượng
    texture_image.save(str(Path(out_dir) / f"{stem}_texture_debug.png"))
    
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture_image,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode='MASK',
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
