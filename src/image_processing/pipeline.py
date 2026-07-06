"""
image_processing/pipeline.py
Pipeline tong hop: Chay toan bo 12 buoc xu ly anh theo thu tu CVIP.

Pipeline:
  Buoc 1:  Chuyen doi khong gian mau (Color Spaces)
  Buoc 2:  Loc tan so Fourier (Fourier Filtering)
  Buoc 3:  Khu nhieu Wavelet (Wavelet Denoising)
  Buoc 4:  Bien doi hinh hoc (Geometric Transform)
  Buoc 5:  Xu ly hinh thai hoc (Morphology)
  Buoc 6:  Phat hien canh (Edge Detection)
  Buoc 7:  Phan doan anh (Segmentation)
  Buoc 8:  Phat hien dac trung (Feature Detection)
  Buoc 9:  Doi sanh dac trung (Feature Matching) -- bo qua neu chi co 1 anh
  Buoc 10: Uoc luong Depth (Depth Estimation)
  Buoc 11: Tao Point Cloud
  Buoc 12: Dung Mesh 3D (TripoSR / Poisson)
"""

import sys
import time
import importlib
import traceback
from pathlib import Path

import cv2
import numpy as np


def _import_module(name):
    """Import module bang importlib (ho tro ten file bat dau bang so)."""
    return importlib.import_module(f"src.image_processing.{name}")


def run_full_pipeline(
    image_path: str,
    output_dir: str,
    use_triposr: bool = True,
    save_intermediate: bool = True,
    on_mesh_ready = None,
    on_depth_ready = None,
) -> dict:
    """
    Chay toan bo pipeline 12 buoc tu anh 2D -> mo hinh 3D.

    Args:
        image_path: Duong dan anh dau vao.
        output_dir: Thu muc luu ket qua.
        use_triposr: True = dung TripoSR dung mesh. False = chi Poisson.
        save_intermediate: True = luu ket qua tung buoc.

    Returns:
        dict chua ket qua tung buoc va duong dan file cuoi cung.
    """
    from src.image_processing.utils import load_image, save_image, show_images

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "steps": {},
        "model_path": None,
        "point_cloud_path": None,
        "errors": [],
    }

    img = load_image(image_path)
    stem = Path(image_path).stem
    start_total = time.time()

    print("=" * 60)
    print("  PIPELINE 3D OBJECT RECONSTRUCTION")
    print(f"  Anh: {image_path}")
    
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  [HARDWARE] Đã phát hiện GPU: {gpu_name}")
        print("  [HARDWARE] Sẽ sử dụng GPU để tăng tốc tối đa quá trình tạo 3D!")
    else:
        print("  [HARDWARE] Không phát hiện GPU. Sẽ chạy trên CPU (có thể mất nhiều thời gian hơn).")
        
    print("=" * 60)

    # == Buoc 1: Chuyen doi khong gian mau ==
    try:
        print("\n[1/12] Chuyen doi khong gian mau...")
        t = time.time()
        mod = _import_module("01_color_spaces")

        gray = mod.convert_to_grayscale(img)
        hsv = mod.convert_to_hsv(img)
        equalized = mod.histogram_equalization(img)

        results["steps"]["01_color"] = {
            "gray_shape": gray.shape,
            "time": time.time() - t,
        }

        if save_intermediate:
            show_images(
                [img, gray, hsv, equalized],
                ["Goc", "Grayscale", "HSV", "CLAHE"],
                save_path=str(out / f"{stem}_01_color.png"),
            )
        print(f"  OK ({time.time() - t:.2f}s)")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 1: {e}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = img

    # == Buoc 2: Loc Fourier ==
    try:
        print("\n[2/12] Loc tan so Fourier...")
        t = time.time()
        mod = _import_module("02_fourier_filtering")

        _, spectrum = mod.apply_fft(gray)
        denoised_fourier = mod.lowpass_filter(gray, radius=50)

        results["steps"]["02_fourier"] = {"time": time.time() - t}

        if save_intermediate:
            show_images(
                [gray, spectrum, denoised_fourier],
                ["Grayscale", "FFT Spectrum", "Low-pass Filtered"],
                save_path=str(out / f"{stem}_02_fourier.png"),
            )
        print(f"  OK ({time.time() - t:.2f}s)")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 2: {e}")

    # == Buoc 3: Khu nhieu Wavelet ==
    try:
        print("\n[3/12] Khu nhieu Wavelet...")
        t = time.time()
        mod = _import_module("03_wavelet_denoising")

        denoised_wavelet = mod.wavelet_denoise(gray, wavelet="db4", level=2)
        LL, LH, HL, HH = mod.wavelet_decompose(gray)

        results["steps"]["03_wavelet"] = {"time": time.time() - t}

        if save_intermediate:
            show_images(
                [gray, denoised_wavelet, LL, HH],
                ["Goc", "Wavelet Denoised", "LL (Approx)", "HH (Diagonal)"],
                save_path=str(out / f"{stem}_03_wavelet.png"),
            )
        print(f"  OK ({time.time() - t:.2f}s)")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 3: {e}")

    # == Buoc 4: Bien doi hinh hoc ==
    try:
        print("\n[4/12] Bien doi hinh hoc...")
        t = time.time()
        mod = _import_module("04_geometric_transform")

        # Đã loại bỏ phép xoay 15 độ vì làm sai lệch cấu trúc 3D
        resized = mod.resize_image(img, width=400)

        results["steps"]["04_geometric"] = {"time": time.time() - t}

        if save_intermediate:
            show_images(
                [img, resized],
                ["Goc", "Resize 400px"],
                save_path=str(out / f"{stem}_04_geometric.png"),
            )
        print(f"  OK ({time.time() - t:.2f}s)")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 4: {e}")

    # == Buoc 5: Hinh thai hoc ==
    mask = None
    try:
        print("\n[5/12] Xu ly hinh thai hoc...")
        t = time.time()
        mod = _import_module("05_morphology")

        mask = mod.create_clean_mask(img)
        grad = mod.morphological_gradient(mask)

        results["steps"]["05_morphology"] = {"time": time.time() - t}

        if save_intermediate:
            show_images(
                [img, mask, grad],
                ["Goc", "Clean Mask", "Morph Gradient"],
                save_path=str(out / f"{stem}_05_morphology.png"),
            )
        print(f"  OK ({time.time() - t:.2f}s)")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 5: {e}")

    # == Buoc 6: Phat hien canh ==
    try:
        print("\n[6/12] Phat hien canh...")
        t = time.time()
        mod = _import_module("06_edge_detection")

        canny = mod.canny_edge(img, 100, 200)
        sobel = mod.sobel_edge(img)
        laplacian = mod.laplacian_edge(img)

        results["steps"]["06_edge"] = {"time": time.time() - t}

        if save_intermediate:
            show_images(
                [gray, canny, sobel, laplacian],
                ["Grayscale", "Canny", "Sobel", "Laplacian"],
                save_path=str(out / f"{stem}_06_edge.png"),
            )
        print(f"  OK ({time.time() - t:.2f}s)")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 6: {e}")

    # == Buoc 7: Phan doan anh ==
    try:
        print("\n[7/12] Phan doan anh...")
        t = time.time()
        mod = _import_module("07_segmentation")

        otsu = mod.otsu_threshold(img)
        kmeans = mod.kmeans_segment(img, k=4)

        results["steps"]["07_segmentation"] = {"time": time.time() - t}

        if save_intermediate:
            show_images(
                [img, otsu, kmeans],
                ["Goc", "Otsu Threshold", "K-Means (K=4)"],
                save_path=str(out / f"{stem}_07_segmentation.png"),
            )
        print(f"  OK ({time.time() - t:.2f}s)")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 7: {e}")

    # == Buoc 8: Phat hien dac trung ==
    try:
        print("\n[8/12] Phat hien dac trung...")
        t = time.time()
        mod = _import_module("08_feature_detection")

        kp_sift, des_sift = mod.detect_sift(img, mask=mask)
        kp_orb, des_orb = mod.detect_orb(img, mask=mask)

        img_sift = mod.draw_keypoints(img, kp_sift, color=(0, 255, 0))
        img_orb = mod.draw_keypoints(img, kp_orb, color=(255, 0, 0))

        results["steps"]["08_features"] = {
            "sift_count": len(kp_sift),
            "orb_count": len(kp_orb),
            "time": time.time() - t,
        }

        if save_intermediate:
            show_images(
                [img, img_sift, img_orb],
                [
                    "Goc",
                    f"SIFT ({len(kp_sift)} kp)",
                    f"ORB ({len(kp_orb)} kp)",
                ],
                save_path=str(out / f"{stem}_08_features.png"),
            )
        print(f"  OK: SIFT={len(kp_sift)} kp, ORB={len(kp_orb)} kp ({time.time() - t:.2f}s)")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 8: {e}")

    # == Buoc 9: Doi sanh dac trung (bo qua neu 1 anh) ==
    print("\n[9/12] Doi sanh dac trung...")
    print("  >> Bo qua (can >=2 anh multi-view, demo dung 1 anh)")
    results["steps"]["09_matching"] = {"skipped": True, "reason": "single image"}

    # == Tiền xử lý Tối ưu: Bounding Box Cropping ==
    processed_image_path = image_path
    cropped_mask = mask
    try:
        print("\n[OPTIMIZATION] Bounding Box Cropping (Tập trung AI vào vật thể)...")
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                pad = 20
                H, W = img.shape[:2]
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
                
                img_cropped = img[y1:y2, x1:x2]
                cropped_mask = mask[y1:y2, x1:x2]
                
                processed_image_path = str(out / f"{stem}_cropped.png")
                cv2.imwrite(processed_image_path, img_cropped)
                print(f"  OK: Đã cắt ảnh từ {img.shape[:2]} -> {img_cropped.shape[:2]}")
    except Exception as e:
        print(f"  LOI Crop: {e}")

    # == Buoc 10: Uoc luong Depth ==
    depth_float = None
    try:
        print("\n[10/12] Uoc luong Depth (Depth-Anything-V2)...")
        depth_file_path = out / f"{stem}_depth.png"
        print(f"  >> Đường dẫn ước lượng ảnh depth: {depth_file_path} (chỉ hiển thị, không tải/chạy AI)")
        
        results["steps"]["10_depth"] = {
            "skipped": True
        }
        
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 10: {e}")

    # == Buoc 11: Tao Point Cloud ==
    try:
        print("\n[11/12] Tao Point Cloud...")
        t = time.time()
        mod = _import_module("11_point_cloud")

        if depth_float is not None:
            img_to_use = cv2.imread(processed_image_path) if processed_image_path != image_path else img
            pcd = mod.depth_to_pointcloud(depth_float, img_to_use)
            info = mod.pointcloud_info(pcd)

            # ply_path = str(out / f"{stem}.ply")
            # mod.save_pointcloud(pcd, ply_path)
            # results["point_cloud_path"] = ply_path

            results["steps"]["11_pointcloud"] = {
                "num_points": info["num_points"],
                "time": time.time() - t,
            }
            print(f"  OK: {info['num_points']} diem ({time.time() - t:.2f}s) (không lưu file .ply)")
        else:
            print("  >> Bo qua (khong co depth map)")
            results["steps"]["11_pointcloud"] = {"skipped": True}

    except ImportError:
        print("  LOI: Can cai open3d: pip install open3d")
        results["errors"].append("Step 11: open3d not installed")
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 11: {e}")

    # == Buoc 12: Dung Mesh 3D ==
    try:
        print("\n[12/12] Dung Mesh 3D...")
        t = time.time()
        mod = _import_module("12_mesh_reconstruction")

        if use_triposr:
            glb_path, raw_mesh, model, scene_code = mod.reconstruct_triposr(
                processed_image_path, str(out), mc_resolution=256, foreground_ratio=0.95
            )
            results["model_path"] = glb_path
            results["steps"]["12_mesh"] = {
                "method": "TripoSR",
                "path": glb_path,
                "time": time.time() - t,
            }
            print(f"  OK: Mesh GLB: {glb_path} ({time.time() - t:.2f}s)")
            
            if on_mesh_ready:
                try:
                    on_mesh_ready(glb_path)
                except Exception as e:
                    print(f"  Loi khi goi callback giao dien: {e}")
        else:
            print("  >> TripoSR tat. Thu Poisson...")
            if depth_float is not None:
                mod_pc = _import_module("11_point_cloud")
                pcd = mod_pc.depth_to_pointcloud(depth_float, img)
                mesh = mod.poisson_reconstruction(pcd, depth=8)
                mesh = mod.smooth_mesh(mesh)
                mesh_path = str(out / f"{stem}_mesh.ply")
                mod.save_mesh(mesh, mesh_path)
                results["model_path"] = mesh_path
                results["steps"]["12_mesh"] = {
                    "method": "Poisson",
                    "path": mesh_path,
                    "time": time.time() - t,
                }
    except Exception as e:
        print(f"  LOI: {e}")
        results["errors"].append(f"Step 12: {e}")
        traceback.print_exc()

    # == Buoc 13: UV Mapping & Texture ==
    if use_triposr and 'raw_mesh' in locals():
        try:
            print("\n[13/13] UV Mapping & Texture...")
            t = time.time()
            mod_uv = _import_module("13_uv_mapping")
            textured_glb_path = mod_uv.apply_uv_mapping(
                raw_mesh, model, scene_code, str(out), stem, texture_res=1024
            )
            results["model_path"] = textured_glb_path
            results["steps"]["13_uv_mapping"] = {
                "path": textured_glb_path,
                "time": time.time() - t
            }
            print(f"  OK ({time.time() - t:.2f}s)")
        except Exception as e:
            print(f"  LOI: {e}")
            results["errors"].append(f"Step 13: {e}")

    # == Tong ket ==
    total_time = time.time() - start_total
    results["total_time"] = total_time

    print("\n" + "=" * 60)
    print(f"  PIPELINE HOAN TAT ({total_time:.1f}s)")
    if results["model_path"]:
        print(f"  Mo hinh 3D: {results['model_path']}")
    if results["point_cloud_path"]:
        print(f"  Point cloud: {results['point_cloud_path']}")
    if results["errors"]:
        print(f"  Canh bao: Co {len(results['errors'])} loi:")
        for err in results["errors"]:
            print(f"    - {err}")
    print("=" * 60)

    return results


# Ham tuong thich voi demo/app.py cu
def run_pipeline(image_path, output_dir, **kwargs) -> Path:
    """
    Wrapper tuong thich voi giao dien cu (demo/app.py goi ham nay).
    Tra ve Path den file GLB.
    """
    results = run_full_pipeline(
        str(image_path), str(output_dir),
        use_triposr=True, save_intermediate=False,
        **kwargs
    )

    if results["model_path"]:
        return Path(results["model_path"])
    else:
        raise RuntimeError(
            "Pipeline khong tao duoc mo hinh 3D. "
            f"Loi: {results['errors']}"
        )


# ====================================================================
#  DEMO -- Chay: python -m src.image_processing.pipeline
# ====================================================================
if __name__ == "__main__":
    import argparse
    from src.image_processing.utils import get_sample_image_path
    from src.image_processing import OUTPUT_DIR

    parser = argparse.ArgumentParser(description="3D Object Reconstruction Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Duong dan anh dau vao")
    parser.add_argument("--output", type=str, default=None, help="Thu muc output")
    parser.add_argument("--no-triposr", action="store_true", help="Khong dung TripoSR")
    parser.add_argument("--no-save", action="store_true", help="Khong luu ket qua trung gian")
    args = parser.parse_args()

    img_path = args.image or get_sample_image_path()
    out_dir = args.output or str(OUTPUT_DIR / "pipeline_results")

    results = run_full_pipeline(
        img_path, out_dir,
        use_triposr=not args.no_triposr,
        save_intermediate=not args.no_save,
    )
