# 👟 TÁI TẠO VẬT THỂ 3D TỪ ẢNH 2D ĐƠN

### Zero-shot Single-image 3D Object Reconstruction

![Python](https://img.shields.io/badge/Python-3.11.x-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Streamlit-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 👥 1. Thành viên nhóm

| Họ và Tên            | Vai trò              | Nhiệm vụ chính                                                          |
| :------------------- | :------------------- | :---------------------------------------------------------------------- |
| **Lê Minh Huy**      | Leader, Thuyết trình | Quản lý tiến độ dự án, chuẩn bị slide và thuyết trình báo cáo chính     |
| **Nguyễn Tiến Đức**  | Code                 | Thiết kế hệ thống, lập trình pipeline AI chính, xử lý toán học & đồ họa |
| **Bùi Văn Ý**        | Code                 | Hỗ trợ lập trình các tính năng, tối ưu hóa hiệu năng Streamlit          |
| **Dương Bích Tuyền** | Làm báo cáo          | Nghiên cứu cơ sở lý thuyết, soạn thảo báo cáo kỹ thuật                  |
| **Nguyễn Tuyết Nhi** | Làm báo cáo          | Thu thập dữ liệu thử nghiệm, tổng hợp kết quả chạy thực tế              |
| **Trần Phước Lộc**   | Thuyết trình, tester | Chuẩn bị nội dung thuyết trình, demo ứng dụng thực tế                   |

---

## 📖 2. Giới thiệu dự án

### Bài toán đặt ra

Tạo mô hình 3D chất lượng cao theo cách thủ công đòi hỏi **nhiều thời gian** và **kỹ năng chuyên nghiệp** trên các phần mềm phức tạp như Blender, Maya, ZBrush — đây là rào cản lớn với cá nhân và doanh nghiệp vừa nhỏ muốn **số hóa sản phẩm nhanh**.

### Hướng giải quyết

> Xây dựng hệ thống **AI tự động** biến một bức ảnh 2D thành mô hình 3D hoàn chỉnh theo hướng **Zero-shot** — không cần huấn luyện lại cho từng vật thể mới.

### Ứng dụng thực tiễn

| Lĩnh vực              | Ứng dụng                                             |
| :-------------------- | :--------------------------------------------------- |
| Thương mại điện tử    | Khách hàng xem sản phẩm 360° trước khi mua           |
| Game / Nội dung số    | Rút ngắn thời gian tạo game assets                   |
| AR / VR               | Số hóa nhanh vật thể thật đưa vào môi trường ảo      |
| Thời trang / Giày dép | Tái tạo chi tiết: nếp nhăn vải, dây giày, hoa văn đế |

### Đối tượng thực nghiệm

Đề tài tập trung vào **giày dép** — nhóm vật thể có kết cấu bề mặt phức tạp (nếp nhăn, dây giày, đế giày) đòi hỏi độ chi tiết cao.

---

## 🗺️ 3. Tổng quan Pipeline hệ thống

```
Ảnh 2D đầu vào (PNG/JPG)
        |
        v
+--------------------------------------------------+
|         PRE-PROCESSING (Tiền xử lý)              |
|                                                  |
|  1. Color Spaces    ->  Grayscale, HSV, CLAHE    |
|  2. Fourier Filter  ->  Khử nhiễu tần số cao     |
|  3. Wavelet Denoise ->  PyWavelets db4 level-2   |
|  4. Geometric       ->  Resize về 400px          |
|  5. Morphology      ->  Sinh clean mask          |
|  6. Edge Detection  ->  Canny / Sobel / Laplacian|
|  7. Segmentation    ->  Otsu + K-Means (K=4)     |
|  8. Feature Detect  ->  SIFT + ORB keypoints     |
|           | Bounding Box Crop (tự động từ mask)  |
+--------------------------------------------------+
        |
        v
+--------------------------------------------------+
|           AI INFERENCE (Suy luận AI)             |
|                                                  |
|  10. Depth-Anything-V2  ->  Depth Map (518px)    |
|  11. Point Cloud        ->  Back-project 3D      |
|  12. TripoSR (LRM)      ->  Mesh 3D trắng .glb  |
|  13. UV Mapping + Bake  ->  Mesh màu .glb        |
+--------------------------------------------------+
        |
        v
Đầu ra: 2 file GLB
  - {stem}.glb           <- Mesh trắng (hiển thị tức thì)
  - {stem}_textured.glb  <- Mesh có màu đầy đủ (sản phẩm cuối)
```

---

## 🔧 4. Kiến trúc tiền xử lý (PRE-PROCESSING)

### 4.1 Sơ đồ luồng xử lý ảnh

```
+------------------+
|    Ảnh gốc       |  (PNG/JPG - bất kỳ độ phân giải)
|    (Đầu vào)     |
+--------+---------+
         |
         v
+--------+---------+
|   Tách nền       |  rembg + alpha_matting=True
|   (rembg)        |  -> Giữ viền mảnh: dây giày, đế giày
+--------+---------+
         |
         v
+--------+---------+
|     Crop         |  Bounding Box từ mask (Bước 5)
|   (Auto Crop)    |  -> Padding 20px, tập trung vào vật thể
+--------+---------+
         |
         v
+--------+---------+
|    Resize        |  foreground_ratio = 0.85
|  (Foreground)    |  -> Chủ thể chiếm 85% khung hình
+--------+---------+
         |
         v
+--------+---------+
|   Nền xám        |  Alpha-composite với nền 50% xám trung tính
|   trung tính     |  -> TripoSR không bị lệch tông màu biên
+--------+---------+
         |
         v
+--------+---------+
|    Ảnh sạch      |  Sẵn sàng đưa vào AI
|    (Đầu ra)      |
+------------------+
```

### 4.2 Chi tiết 13 bước xử lý CVIP

|  #  | Bước                       | Thuật toán                             | Mục đích                  |
| :-: | :------------------------- | :------------------------------------- | :------------------------ |
|  1  | Color Spaces               | Grayscale, HSV, CLAHE                  | Chuẩn hóa độ sáng         |
|  2  | Fourier Filter             | FFT + Low-pass (r=50)                  | Khử nhiễu tần số cao      |
|  3  | Wavelet Denoise            | PyWavelets db4, level=2                | Làm mượt bề mặt           |
|  4  | Geometric                  | Resize về 400px                        | Chuẩn hóa đầu vào         |
|  5  | Morphology                 | Open/Close → Clean Mask                | Tạo mask vật thể          |
|  6  | Edge Detection             | Canny(100-200), Sobel, Laplacian       | Phát hiện biên            |
|  7  | Segmentation               | Otsu + K-Means (K=4)                   | Phân vùng màu             |
|  8  | Feature Detect             | SIFT + ORB                             | Phân tích keypoint        |
|  9  | Feature Match              | _(bỏ qua — chế độ một ảnh)_            | —                         |
|  ★  | **Bounding Box Crop**      | Từ mask Bước 5 + padding 20px          | **Tập trung vào vật thể** |
| 10  | **Ước lượng chiều sâu**    | **Depth-Anything-V2** (ViT-S, 518px)   | **Bản đồ chiều sâu**      |
| 11  | Point Cloud                | Back-project + Open3D                  | Minh họa học thuật        |
| 12  | **Dựng Mesh 3D (TripoSR)** | **LRM + Marching Cubes (256³)**        | **Dựng khối 3D**          |
| 13  | **UV Mapping + Texture**   | **xatlas + triplane bake (2048×2048)** | **Phủ màu hoàn chỉnh**    |

---

## 🤖 5. Các mô hình AI sử dụng

### 5.1 Depth-Anything-V2 — Ước lượng chiều sâu

```
Ảnh RGB (518×518)
      |
      v
[Backbone] Vision Transformer encoder (ViT-S)
      |  Trích xuất đặc trưng đa tỉ lệ
      v
[Head] DPT Decoder
      |  Hồi quy bản đồ chiều sâu mật độ pixel
      v
Depth Map [0, 1]  ->  Dùng để dựng Point Cloud
```

- **Kiến trúc:** Vision Transformer (ViT-Small)
- **Độ phân giải suy luận:** 518px
- **Chế độ:** Zero-shot inference (không train lại)
- **Đầu ra:** Bản đồ chiều sâu chuẩn hóa [0, 1]

---

### 5.2 TripoSR — Large Reconstruction Model (LRM)

```
Ảnh đã tách nền (RGBA)
      |
      v
[Backbone] Image-to-Triplane Transformer Encoder
      |  Mã hóa ảnh -> biểu diễn Tri-plane (3 mặt phẳng trực giao)
      v
[Head] MLP Decoder
      |  Dự đoán Neural Implicit Field (SDF + màu tại tọa độ 3D bất kỳ)
      v
Marching Cubes (resolution = 256³)
      |  Trích xuất mesh tại SDF = 0
      v
Mesh trắng -> {stem}.glb         (Bước 12, hiển thị tức thì)
      |
      v
xatlas UV Unwrap + Triplane Multi-view Bake (2048×2048)
      |
      v
Mesh màu -> {stem}_textured.glb  (Bước 13, sản phẩm cuối)
```

| Thông số         | Giá trị                                        |
| :--------------- | :--------------------------------------------- |
| Kiến trúc        | Large Reconstruction Model (LRM) — Transformer |
| Biểu diễn 3D     | Tri-plane Neural Implicit                      |
| Marching Cubes   | Resolution = 256³                              |
| UV Texture       | xatlas conformal unwrap + 2048×2048 bake       |
| Foreground ratio | 0.85                                           |
| Chunk size       | 8192                                           |
| Chế độ           | Zero-shot inference                            |

---

## 💡 6. Các cơ chế đặc biệt

### 6.1 Progressive Feedback — Hiển thị tiến dần

```
[Bước 12 hoàn thành]  ->  Mesh TRẮNG hiển thị ngay  (người dùng thấy ngay hình khối)
          |
[Bước 13 hoàn thành]  ->  Mesh MÀU cập nhật cuối cùng  (sản phẩm hoàn chỉnh)
```

Người dùng không phải chờ đợi trong màn hình trống — hình khối 3D trắng xuất hiện ngay khi TripoSR hoàn thành, trước khi quá trình UV baking kết thúc.

---

### 6.2 Cache SHA-256 — Tiết kiệm tài nguyên

```
Ảnh đầu vào
    |
    v
Tính mã băm SHA-256
    |
    +--- [Đã có trong cache] --->  Load .glb ngay lập tức (không tốn GPU)
    |
    +--- [Chưa có]  --->  Chạy toàn bộ Pipeline 13 bước  ->  Lưu kết quả vào cache
```

- **Exact match:** SHA-256 là exact matching, không có false positive
- **Tiết kiệm:** Cùng một ảnh → lần sau trả kết quả gần như tức thì
- **Lưu ý:** Cache gần đúng HSV đã **tắt** do false positive — hai giày khác nhau nhưng cùng nền trắng → trả nhầm mô hình

---

### 6.3 Human-in-the-Loop — Thu thập dữ liệu thực tế

```
Người dùng xem kết quả 3D
    |
    +--- Hài lòng  ->  Kết thúc
    |
    +--- Phát hiện lỗi  ->  Nhập mô tả lỗi
              |
              v
         Lưu vào data/feedback/
         +------------------------+
         |  ảnh gốc               |
         |  file .glb lỗi         |  ->  Dữ liệu cho fine-tune tương lai
         |  mô tả lỗi (văn bản)   |
         +------------------------+
```

---

## 📊 7. Dataset & Dữ liệu

| Tiêu chí         | Giá trị                                   |
| :--------------- | :---------------------------------------- |
| Tên dataset      | `dataset_shoe` (tự xây dựng)              |
| Số lượng ảnh     | **192 ảnh** giày                          |
| Độ phân giải     | 1024 × 1024 px                            |
| Định dạng        | RGBA (đã có kênh alpha)                   |
| Mục đích         | Kiểm thử pipeline (100% test set)         |
| Dataset training | Không có — mô hình pretrained (zero-shot) |

> **Vì sao không có train/val/test split?**
> TripoSR và Depth-Anything-V2 được dùng **ở chế độ pretrained sẵn**, không huấn luyện lại. Toàn bộ 192 ảnh dùng 100% cho **kiểm thử** (test).

---

## ⚖️ 8. Ưu điểm & Hạn chế

### ✅ Ưu điểm

| Điểm mạnh                | Mô tả                                                               |
| :----------------------- | :------------------------------------------------------------------ |
| **Tốc độ nhanh**         | Vài giây/ảnh — feed-forward 1 lượt, không cần optimize lặp như NeRF |
| **Zero-shot**            | Không cần dữ liệu training riêng cho vật thể mới                    |
| **Pipeline đầy đủ**      | 13 bước CVIP đầy đủ — học thuật & debug từng giai đoạn              |
| **Cache thông minh**     | SHA-256 exact match — cùng ảnh không mất tài nguyên GPU             |
| **Progressive feedback** | Mesh trắng hiện ngay, mesh màu cập nhật sau                         |
| **Human-in-the-loop**    | Thu thập dữ liệu lỗi tự động → nền tảng fine-tune tương lai         |

### ⚠️ Hạn chế

| Hạn chế                | Nguyên nhân                                                    |
| :--------------------- | :------------------------------------------------------------- |
| **Góc khuất**          | Chỉ 1 ảnh → mặt sau vật thể phải suy đoán (hallucination)      |
| **Chi tiết mỏng mất**  | Dây giày mảnh, lưới thoáng khí bị làm mượt ở resolution 256³   |
| **Phụ thuộc tách nền** | Nếu rembg tách sai biên → sai số lan truyền vào mesh cuối      |
| **Cache HSV tắt**      | False positive: 2 giày khác nhau + nền trắng → correlation cao |

---

## 🔬 9. Cơ sở toán học chính

### 9.1 Mô hình Camera lỗ kim — Dựng Point Cloud

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z(u,v)

Trong đó: fx = fy = max(H, W),  cx = W/2,  cy = H/2
```

Mỗi điểm ảnh (u, v) có chiều sâu Z(u,v) được chiếu ngược thành điểm 3D (X, Y, Z).

### 9.2 Signed Distance Function (SDF)

```
SDF(x) = 0   nếu x nằm trên bề mặt vật thể
SDF(x) > 0   nếu x nằm ngoài vật thể
SDF(x) < 0   nếu x nằm bên trong vật thể
```

Ranh giới vật thể: { x | SDF(x) = 0 }
→ Marching Cubes trích xuất mesh tại đây.

### 9.3 Hàm mất mát TripoSR (training gốc)

```
L_total = L_photo + lambda_mask * L_mask + lambda_eikonal * L_eikonal
```

### 9.4 UV Mapping — LSCM (Least Squares Conformal Maps)

```
C(T) = tích phân_T | dU/dx + i*dU/dy |^2 dA
```

xatlas tối thiểu hóa sai số biến dạng góc khi trải bề mặt 3D → UV Atlas 2D.

---

## 🏗️ 10. Cấu trúc dự án

```
3D-Object-Reconstruction/
├── src/
│   ├── image_processing/
│   │   ├── pipeline.py               <- Pipeline chính 13 bước
│   │   ├── 01_color_spaces.py        <- Bước 1: Chuyển không gian màu
│   │   ├── 02_fourier_filtering.py   <- Bước 2: Lọc Fourier
│   │   ├── 03_wavelet_denoising.py   <- Bước 3: Khử nhiễu Wavelet
│   │   ├── 04_geometric_transform.py <- Bước 4: Biến đổi hình học
│   │   ├── 05_morphology.py          <- Bước 5: Hình thái học + mask
│   │   ├── 06_edge_detection.py      <- Bước 6: Phát hiện cạnh
│   │   ├── 07_segmentation.py        <- Bước 7: Phân đoạn ảnh
│   │   ├── 08_feature_detection.py   <- Bước 8: Phát hiện đặc trưng
│   │   ├── 09_feature_matching.py    <- Bước 9: Đối sánh đặc trưng
│   │   ├── 10_depth_estimation.py    <- Bước 10: Ước lượng chiều sâu
│   │   ├── 11_point_cloud.py         <- Bước 11: Tạo Point Cloud
│   │   ├── 12_mesh_reconstruction.py <- Bước 12: Dựng Mesh 3D (TripoSR)
│   │   └── 13_uv_mapping.py          <- Bước 13: UV Mapping & Texture
│   ├── depth_anything_v2/            <- Mã nguồn Depth-Anything-V2
│   └── tsr/                            # Thuật toán chính TripoSR (Tri-plane Reconstruction)
│       ├── models/                     # Kiến trúc mạng nơ-ron sinh tọa độ và màu sắc
│       │   ├── isosurface.py           # Trích xuất bề mặt (Isosurface extraction)
│       │   ├── nerf_renderer.py        # Bộ kết xuất NeRF (Neural Radiance Fields)
│       │   └── network_utils.py        # Các hàm tiện ích mạng nơ-ron
│       ├── system.py                   # Lớp quản lý chính của mô hình TripoSR
│       ├── bake_texture.py             # Kỹ thuật nướng vân bề mặt (Texture Baking) lên Mesh 3D
│       └── utils.py                    # Các hàm tiện ích hỗ trợ tính toán 3D
├── demo/
│   └── app.py                        <- Giao diện Web Streamlit
│
├── data/
│   ├── cache/                        <- Lưu .glb đã sinh (SHA-256)
│   └── feedback/                     <- Dữ liệu human-in-the-loop
│
├── dataset/dataset_shoe/             <- 192 ảnh giày test (1024×1024)
├── weights/
│   └── depth_anything_v2_vits.pth    <- Trọng số Depth-Anything-V2
├── requirements.txt
└── run.bat                           <- Chạy tự động toàn bộ hệ thống
```

---

## 🚀 11. Hướng dẫn chạy nhanh

```bash
# Chỉ cần 1 lệnh — script tự xử lý venv, cài thư viện, mở browser:
.\run.bat
```

Hoặc thủ công:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
streamlit run demo/app.py
```

---

## 🔭 12. Hướng phát triển tiếp theo

| Hướng                   | Mô tả                                                           |
| :---------------------- | :-------------------------------------------------------------- |
| **Multi-view fallback** | Tích hợp COLMAP + Gaussian Splatting khi có video/nhiều ảnh     |
| **Fine-tune TripoSR**   | Dùng dữ liệu `data/feedback/` để fine-tune trên domain giày dép |
| **Cache HSV cải tiến**  | Kết hợp shape embedding để giảm false positive                  |
| **Tăng Marching Cubes** | Resolution 256³ → 384³ khi có GPU mạnh hơn                      |
| **Ground-truth 3D**     | Quét 3D một số mẫu giày thật để tính Chamfer Distance, IoU      |

---

## 📚 Tài liệu tham khảo chính

- **TripoSR:** Stability AI & Tripo AI — https://github.com/VAST-AI-Research/TripoSR
- **Depth Anything V2:** Yang et al. — https://github.com/DepthAnything/Depth-Anything-V2
- **rembg:** Daniel Gatis — https://github.com/danielgatis/rembg
- **xatlas:** UV unwrapping library — https://github.com/jpcy/xatlas
- **Marching Cubes:** Lorensen & Cline, 1987
- **LSCM:** Lévy et al. — Least Squares Conformal Maps

> **Mã nguồn:** https://github.com/huylm231/3D-Object-Reconstruction

---

_Phát triển bởi nhóm nghiên cứu Thị giác Máy tính (Computer Vision) — Học kỳ 2 năm học 2025–2026_
