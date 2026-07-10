"""
src/pipeline.py
Pipeline trung tam: Anh 2D -> Depth Map -> Point Cloud -> GLB
Ket hop Depth-Anything-V2 va 3DObjectReconstruction.
"""

import sys
import os
from pathlib import Path

# Them duong dan src de TripoSR nhan dien duoc cac module trong config
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_pipeline(image_path: Path, output_dir: Path) -> Path:
    """
    Chay toan bo pipeline tu anh 2D sang mo hinh 3D.

    Args:
        image_path: Duong dan den anh dau vao (PNG/JPG).
        output_dir: Thu muc dau ra de luu file .glb / .obj.

    Returns:
        Path den file mo hinh 3D da tao.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # ── Buoc 1: Depth Estimation ──────────────────────────────────────
    depth_map = _estimate_depth(image_path, output_dir)

    # ── Buoc 2: 3D Reconstruction ─────────────────────────────────────
    model_path = _reconstruct_3d(image_path, depth_map, output_dir, stem)

    return model_path


def _estimate_depth(image_path: Path, output_dir: Path):
    """
    Chay Depth-Anything-V2 tren anh dau vao.
    Tra ve numpy array (H, W) chua gia tri do sau.
    """
    try:
        import cv2
        import torch
        import numpy as np
        from depth_anything_v2.dpt import DepthAnythingV2

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Config encoder nhe nhat (vits) de chay tren CPU
        encoder = "vits"
        cfg = {"encoder": encoder, "features": 64, "out_channels": [48, 96, 192, 384]}

        model = DepthAnythingV2(**cfg)
        root = Path(__file__).parent.parent
        checkpoint_candidates = [
            root / "models" / f"depth_anything_v2_{encoder}.pth",
            root / "weights" / f"depth_anything_v2_{encoder}.pth",
        ]
        checkpoint = next((p for p in checkpoint_candidates if p.exists()), None)
        if checkpoint is None:
            raise FileNotFoundError(
                "Checkpoint Depth-Anything-V2 không tồn tại. Kiểm tra thư mục models/ hoặc weights/.\n"
                f"Candidates:\n  - {checkpoint_candidates[0]}\n  - {checkpoint_candidates[1]}\n"
                "Tải về tại: https://huggingface.co/depth-anything/Depth-Anything-V2-Small"
            )

        model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
        model = model.to(device).eval()

        raw = cv2.imread(str(image_path))
        depth = model.infer_image(raw, 518)

        # Chuan hoa ve [0, 255]
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_u8 = (depth_norm * 255).astype("uint8")
        cv2.imwrite(str(output_dir / f"{image_path.stem}_depth.png"), depth_u8)

        return depth_norm  # float [0,1]

    except Exception as e:
        print(f"[PIPELINE] Depth estimation loi: {e}")
        print("[PIPELINE] Dung depth gia dinh (all-zeros)")
        import numpy as np
        return None


def _reconstruct_3d(
    image_path: Path,
    depth_map,   # numpy float [0,1] hoac None
    output_dir: Path,
    stem: str,
) -> Path:
    """
    1. Tao point cloud tu depth map roi xuat ra .ply (phuc vu bao cao hoc thuat).
    2. Su dung TripoSR de dung mesh 3D hoan chinh, xuat file GLB.
    """
    glb_out = output_dir / f"{stem}.glb"
    ply_out = output_dir / f"{stem}.ply"

    # ── Buoc 1: Thu dung thu vien Open3D tao Point Cloud de chung minh co dung Depth ──
    if depth_map is not None:
        try:
            import open3d as o3d
            import numpy as np
            import cv2

            img_bgr = cv2.imread(str(image_path))
            h, w = img_bgr.shape[:2]
            fx = fy = max(h, w)
            cx, cy = w / 2, h / 2

            # Tao depth image tu mang float
            depth_o3d = o3d.geometry.Image((depth_map * 1000).astype("uint16"))
            color_o3d = o3d.geometry.Image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            )

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1000.0,
                depth_trunc=10.0,
                convert_rgb_to_intensity=False,
            )

            intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
            pcd.estimate_normals()

            # Xuat PLY de dung cho bao cao hoc thuat
            o3d.io.write_point_cloud(str(ply_out), pcd)
            print(f"[PIPELINE] Da tao point cloud (bao cao hoc thuat): {ply_out}")

        except ImportError:
            print("[PIPELINE] open3d chua cai dat. Bo qua buoc tao Point Cloud.")
        except Exception as e:
            print(f"[PIPELINE] Point Cloud loi: {e}")

    # ── Buoc 2: Dung TripoSR de tao ra file GLB that su (chuan Object Identity) ──
    try:
        print("[PIPELINE] Dang goi TripoSR de tao Mesh 3D...")

        import torch
        import rembg
        from PIL import Image
        import numpy as np
        
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Khoi tao mo hinh TripoSR
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(8192)
        model.to(device)

        # Xoa phong va can chinh chu the (Foreground Extraction)
        rembg_session = rembg.new_session()
        input_image = Image.open(image_path)
        # Dung alpha_matting de tranh bi cat lem de giay qua sat vi ti le rac roi
        image = remove_background(input_image, rembg_session, alpha_matting=True)
        # Thu nho giay lai mot chut (0.80 thay vi 0.85) de AI hieu ro khoang khong o de giay
        image = resize_foreground(image, 0.80)
        
        # Luu anh tach nen de sau nay app.py luu vao feedback
        nobg_out = output_dir / f"{stem}_nobg.png"
        image.save(str(nobg_out))
        
        # Chuyen ve dinh dang TripoSR can
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
        image_pil = Image.fromarray((image_np * 255.0).astype(np.uint8))

        # Du doan 3D tu 1 anh
        with torch.no_grad():
            scene_codes = model([image_pil], device=device)

        # Trich xuat Mesh voi do phan giai cao (384 thay vi 256 de mau sac bam dung vao vi tri)
        meshes = model.extract_mesh(scene_codes, True, resolution=384)
        mesh = meshes[0]
        
        # Xu ly hien tuong trong suot: Go bo hoac dat muc Alpha len 255 (Max Opaque) cho vertex colors
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = np.array(mesh.visual.vertex_colors)
            if colors.shape[1] == 4:
                # Ép dải Alpha (độ trong suốt) thành đục hoàn toàn (255)
                colors[:, 3] = 255
                mesh.visual.vertex_colors = colors
        
        # Xuat ra file GLB thuc te
        mesh.export(str(glb_out))
        print(f"[PIPELINE] Da xuat file GLB hoan chinh: {glb_out}")
        
        return glb_out

    except Exception as e:
        import traceback
        print(f"\n[PIPELINE - LỖI NGHIÊM TRỌNG] TripoSR Reconstruction thất bại: {e}")
        print("Vui lòng kiểm tra xem đã cài đặt đủ các thư viện trong requirements.txt (torch, transformers, torchmcubes...) chưa.")
        traceback.print_exc()
        
        raise RuntimeError(f"Khong the tao mo hinh 3D do loi: {e}")
