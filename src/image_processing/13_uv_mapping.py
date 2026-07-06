import numpy as np
from pathlib import Path
from PIL import Image

def apply_uv_mapping(mesh, model, scene_code, out_dir, stem, texture_res=1024):
    """
    Áp dụng thuật toán LSCM UV Unwrapping (qua xatlas) và Bake màu thực tế của AI lên lưới 3D.
    """
    import trimesh
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from tsr.bake_texture import bake_texture
    
    # Tạo lưới atlas 2D và render màu
    bake_result = bake_texture(mesh, model, scene_code, texture_res)
    
    # Trích xuất dữ liệu từ bake_result
    vmapping = bake_result["vmapping"]
    indices = bake_result["indices"]
    uvs = bake_result["uvs"]
    colors = bake_result["colors"] # Float32 [0, 1] shape (H, W, 4)
    
    # Khởi tạo mesh mới dựa trên chỉ mục đã unwrap của xatlas
    new_vertices = mesh.vertices[vmapping]
    new_faces = indices
    clean_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False,
    )
    clean_mesh.fix_normals()
    
    # Chuyển đổi mảng màu Float32 sang ảnh PIL để làm Texture
    colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    texture_image = Image.fromarray(colors_uint8, mode="RGBA")
    
    material = trimesh.visual.material.PBRMaterial(
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE'
    )
    clean_mesh.visual = trimesh.visual.TextureVisuals(
        uv=uvs, 
        image=texture_image, 
        material=material
    )
    
    glb_path = Path(out_dir) / f"{stem}_textured.glb"
    clean_mesh.export(str(glb_path), file_type='glb')
    
    return str(glb_path)
