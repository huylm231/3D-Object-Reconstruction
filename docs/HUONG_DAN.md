# docs/HUONG_DAN.md
# Huong Dan Su Dung - 3D Shoe Reconstruction

## Cau truc du an

```
3D-Object-Reconstruction/
|
|-- Depth-Anything-V2/      # Module do sau (giu nguyen)
|-- 3DObjectReconstruction/ # Module dung 3D (giu nguyen)
|-- dataset/                # Dataset mau (anh PNG + file GLB mau)
|
|-- src/                    # Logic chinh cua du an
|   |-- __init__.py
|   |-- pipeline.py         # Dieu phoi toan bo pipeline 2D->3D
|
|-- models/                 # AI model wrappers
|   |-- __init__.py
|   |-- depth_wrapper.py    # Wrapper cho Depth-Anything-V2
|
|-- demo/                   # Server chay thu
|   |-- server.py           # Server HTTP thuan Python (khong can HTML rieng)
|
|-- docs/                   # Tai lieu
|   |-- HUONG_DAN.md        # File nay
|
|-- data/
|   |-- uploads/            # Anh nguoi dung tai len
|   |-- outputs/            # Mo hinh 3D dau ra (.glb, .ply, .obj, _nobg.png)
|   |-- cache/              # Noi luu model de load nhanh
|   |-- feedback/           # Noi luu bao cao loi cua nguoi dung
|-- .venv/                  # Moi truong ao Python 3.11.9
|-- requirements.txt        # Thu vien phu thuoc
|-- README.md
```

## Cai dat

### 1. Tao moi truong ao
```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
```

### 2. Cai thu vien (cho pipeline AI day du)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python open3d
```

> **Luu y:** Server co the chay KHONG can torch/opencv nho co GLB mau trong dataset/.

### 3. Tai checkpoint Depth-Anything-V2 (tuy chon)
```
Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth
```
Link tai: https://huggingface.co/depth-anything/Depth-Anything-V2-Small

## Chay ung dung

```bash
streamlit run demo/app.py
```

Trang web sẽ tự động mở trên trình duyệt (thường ở http://localhost:8501)

## Cach hoat dong

```
Nguoi dung -> Upload anh JPG/PNG
     |
     v
demo/server.py (nhan POST /api/upload)
     |
     v
src/pipeline.py
     |-- _estimate_depth()    <- models/depth_wrapper.py <- Depth-Anything-V2/
     |-- _reconstruct_3d()    <- Open3D point cloud + 3DObjectReconstruction/
     v
outputs/<job_id>.glb
     |
     v
GET /outputs/<job_id>.glb -> model-viewer hien thi 3D
```

## Che do Fallback (khi chua co AI model)

Neu chua tai checkpoint hoac chua cai torch, server van chay binh thuong:
- Pipeline se gap loi o buoc depth estimation
- Server tu dong dung file GLB mau: `dataset/nike_air_zoom_pegasus_36.glb`
- Nguoi dung van thay mo hinh 3D (cua mau) trong trinh duyet
