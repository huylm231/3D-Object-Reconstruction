# 👟 Dự Án Tái Tạo Mô Hình 3D (3D Object Reconstruction)

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python Version](https://img.shields.io/badge/Python-3.11.9-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Streamlit-orange)

---

## 📖 1. Giới thiệu tổng quan
Dự án **3D Object Reconstruction** (Tái tạo vật thể 3D) là một hệ thống Trí tuệ Nhân tạo (AI) chuyên sâu, tập trung giải quyết bài toán biến đổi **một bức ảnh 2D duy nhất thành mô hình 3D hoàn chỉnh** (Zero-shot Single-image 3D Reconstruction), với sự tập trung đặc biệt vào các vật thể phức tạp như giày dép (yêu cầu giữ lại chi tiết nếp nhăn, vân vải).

Trong thực tế, việc thiết kế mô hình 3D thủ công đòi hỏi rất nhiều thời gian và kỹ năng thiết kế chuyên nghiệp (sử dụng Blender, Maya, ZBrush...). Hệ thống này tự động hóa hoàn toàn quy trình này bằng việc kết hợp các công nghệ AI tiên tiến (SDF, Tri-plane, Transformers) để tách nền ảnh, ước lượng không gian, đúc khối lưới đa giác (Mesh) và phủ màu, tạo thành một file chuẩn `.glb` có thể tương tác 360 độ ngay trên trình duyệt Web.

### ✨ Các tính năng nổi bật:
- **Tái tạo 3D tức thì (Fast-Demo):** Chỉ cần một bức ảnh, không cần train lại model, tạo ra vật thể 3D trong vài giây.
- **Tiền xử lý tự động (Auto Pre-processing):** Tự động nhận diện vật thể, tách nền và căn chỉnh (Crop) tối ưu hóa góc độ.
- **Cơ chế Human-in-the-loop:** Hệ thống cho phép người dùng đánh giá và báo cáo lỗi mô hình. Các báo cáo này tự động được lưu trữ làm bộ dữ liệu (Dataset) để Fine-tune AI trong tương lai.
- **Cơ chế Caching thông minh:** Các ảnh đã xử lý được băm (hash) mã SHA-256 để lưu trữ kết quả, giảm thiểu thời gian chờ ở những lần xử lý sau.

---

## 📂 2. Cấu trúc thư mục (Architecture Directory)

Dự án đã được quy hoạch chuẩn chỉ theo cấu trúc phần mềm hiện đại, phân tách rõ ràng giữa mã nguồn, trọng số mô hình và tài liệu:

```text
3D-Object-Reconstruction/
│
├── data/                       # Dữ liệu vào/ra trong quá trình chạy ứng dụng
│   ├── cache/                  # Lưu trữ mô hình 3D (.glb) đã tạo thành công
│   ├── feedback/               # Lưu các báo cáo lỗi từ người dùng (dataset tương lai)
│   ├── outputs/                # Nơi xuất các file tạm thời
│   └── uploads/                # Ảnh do người dùng tải lên
│
├── dataset/                    # Chứa dataset mẫu và các file .glb ví dụ
│
├── demo/
│   └── app.py                  # Mã nguồn giao diện Web (Streamlit), xử lý UI
│
├── docs/                       # Hệ thống tài liệu kỹ thuật và lý thuyết
│   ├── theory/                 # Tài liệu học thuật chuyên sâu (.pdf)
│   ├── ly_thuyet_ap_dung.md    # Lý thuyết toán học & AI đã ứng dụng
│   ├── 3D_RECONSTRUCTION_TECHNICAL_GUIDE.md # Hướng dẫn kỹ thuật
│   └── CVIP_Kien_Thuc_Cot_Loi_3D_Reconstruction.md
│
├── src/                        # Thư mục mã nguồn Cốt lõi (AI Pipeline)
│   ├── image_processing/       # Các thuật toán xử lý ảnh (khử nhiễu, tách nền,...)
│   ├── models/                 # Các wrapper kết nối model AI
│   ├── depth_anything_v2/      # Mã nguồn ước lượng chiều sâu
│   ├── tsr/                    # Thuật toán lõi TripoSR 
│   └── pipeline.py             # Luồng xử lý chính: Ảnh -> Xoá phông -> AI -> GLB
│
├── weights/                    # Chứa các file trọng số (Weights/Checkpoints) của AI
│   └── depth_anything_v2_vits.pth
│
├── .venv/                      # Môi trường ảo Python (được tạo tự động)
├── run.bat                     # Script khởi động tự động cho Windows
├── requirements.txt            # Danh sách các thư viện cần thiết
└── README.md                   # Tài liệu bạn đang đọc
```

---

## 🚀 3. Hướng dẫn Cài đặt & Khởi chạy

### 💻 Yêu cầu hệ thống (System Requirements)
- **Hệ điều hành:** Khuyến nghị Windows 10/11.
- **Python:** Phiên bản chuẩn 3.11.x (Hệ thống đã ép cứng tương thích với `numpy < 2.0.0`).
- **Phần cứng:** Ưu tiên máy tính có GPU NVIDIA (Card rời) để quá trình nội suy 3D diễn ra nhanh chóng. Nếu không có GPU, hệ thống tự động fallback về chạy bằng CPU.

### Phương pháp 1: Khởi chạy Tự động bằng Script (Khuyên dùng)
Dự án được tích hợp sẵn file thực thi `.bat` giúp tự động hóa toàn bộ quá trình thiết lập. Bạn chỉ cần:
1. Mở Terminal (PowerShell hoặc CMD) tại thư mục dự án.
2. Gõ lệnh:
   ```bash
   .\run.bat
   ```
> **Điều gì xảy ra khi chạy `run.bat`?**
> Nó sẽ quét máy tính, tự động tạo môi trường ảo `.venv`, tự tải `PyTorch` phiên bản CUDA tương thích, cài đặt các thư viện từ `requirements.txt`, và tự động mở trình duyệt web lên trang giao diện của ứng dụng.

### Phương pháp 2: Cài đặt và Chạy Thủ công
Nếu bạn muốn tự tay quản lý môi trường:
```bash
# 1. Tạo môi trường ảo (ưu tiên Python 3.11)
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/Mac:
source .venv/bin/activate

# 3. Cài đặt PyTorch (Sử dụng CUDA 11.8 cho GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 5. Khởi chạy Giao diện Web
streamlit run demo/app.py
```

---

## ⚙️ 4. Luồng xử lý kỹ thuật (Technical Pipeline)

Hệ thống AI xử lý một tấm ảnh theo quy trình khép kín (`src/pipeline.py`):

1. **Upload & Hashing:** Giao diện nhận ảnh từ người dùng. Hệ thống băm mã SHA-256 của ảnh để tra cứu trong thư mục `cache/`. Nếu đã từng chạy, file GLB sẽ được load ngay lập tức để tiết kiệm tài nguyên GPU.
2. **Pre-processing (Tiền xử lý):**
   - Sử dụng thư viện `rembg` để cắt nền vật thể. 
   - Đã được tối ưu với tham số `alpha_matting=True` và `resize_foreground=0.80` giúp các vật thể có chi tiết phức tạp (đế giày, dây giày) không bị lẹm.
   - Ứng dụng xử lý hình thái học (Morphology) để xác định Bounding Box, ép vật thể chiếm trọn khung hình.
3. **Geometry Extraction:** Mô hình ước lượng chiều sâu (`Depth-Anything-V2` kiến trúc Transformer) tính toán độ xa gần của từng pixel.
4. **3D Generation (Dựng khối 3D):** Thuật toán LRM (Large Reconstruction Model - TripoSR) dự đoán trường Neural Implicit (SDF/Tri-plane) từ ảnh. Sau đó, thuật toán **Marching Cubes** trích xuất lưới 3D với độ phân giải cao (`resolution=384`).
5. **Post-processing (Hậu xử lý):** Quét qua cấu trúc Lưới (Mesh) để loại bỏ dải màu Alpha dư thừa (ngăn chặn hiện tượng lỗi vật liệu trong suốt khi đưa lên môi trường 3D Web). Xuất ra file `.glb`.

---

## 🛠 5. Các lỗi thường gặp (Troubleshooting)

Trong quá trình sử dụng, nếu bạn gặp phải các lỗi sau, hãy tham khảo cách xử lý:

| Hiện tượng lỗi | Nguyên nhân gốc rễ | Cách xử lý dứt điểm |
| :--- | :--- | :--- |
| **Báo lỗi "No module named 'cv2' hoặc thư viện khác"** | Môi trường ảo (`.venv`) chưa được kích hoạt hoặc cài thiếu thư viện. | Tắt Terminal hiện tại, chạy lại file `.\run.bat` để script tự động sửa lỗi. |
| **Cài đặt `pip` báo lỗi "Hash mismatch"** | File PyTorch rất lớn (~2.5GB), kết nối mạng thiếu ổn định làm file bị hỏng khi tải. | Chạy lại `.\run.bat` lần nữa. Script đã được cấu hình ép tải trực tiếp từ máy chủ NVIDIA để hạn chế rủi ro. |
| **Lỗi "numpy.ndarray object has no attribute ptp"** | Xung đột phiên bản khi Python (>=3.12) kết hợp với Numpy 2.x. | Hệ thống bắt buộc dùng `numpy < 2.0.0` và Python 3.11. Nếu bạn tự cài thủ công, hãy `pip install numpy==1.26.4`. |
| **Mô hình 3D xuất ra bị "trong suốt" hoặc nhòe màu** | Các Engine Web 3D đọc sai thông số Alpha Channel của Vertex Color. | Lỗi này đã được vá trong `src/pipeline.py` (loại bỏ dải màu thứ 4 của vertex). KHÔNG sửa đổi đoạn code xử lý Alpha trong mã nguồn. |
| **Lỗi biên dịch `torchmcubes` (C++) trên Windows** | Thư viện `torchmcubes` đòi hỏi Microsoft Visual Studio Build Tools, gây lỗi cho 90% máy Windows. | Dự án đã chuyển đổi kiến trúc sang dùng thư viện `PyMCubes` (đã được biên dịch sẵn) để đảm bảo "Plug and Play". |
| **Lỗi không tìm thấy file trọng số `.pth`** | Di chuyển nhầm hoặc xóa nhầm thư mục `weights/`. | Đảm bảo file `depth_anything_v2_vits.pth` luôn nằm trong `weights/`. |

---

## 📚 6. Hệ thống tài liệu chuyên sâu

Để hiểu rõ hơn về các thuật toán hình học, đồ họa máy tính, biến đổi Fourier, thuật toán bọc bề mặt Poisson hoặc SDF, vui lòng truy cập thư mục `docs/`. Trong đó bao gồm:
- **`ly_thuyet_ap_dung.md`**: Giải thích cụ thể các công thức toán học và AI đã ứng dụng.
- **`docs/theory/`**: Các tài liệu khoa học gốc dạng PDF.
- **Kế hoạch dự phòng (Backup Plan)**: Sử dụng Multi-view (COLMAP + Gaussian Splatting) trong trường hợp Zero-shot 1 ảnh không đáp ứng đủ nhu cầu hình khối.

---
*Phát triển bởi đội ngũ nghiên cứu ứng dụng Thị giác Máy tính (Computer Vision) & Trí tuệ Nhân tạo.*
