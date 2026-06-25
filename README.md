# 👟 3D Shoe Reconstruction

## 📖 Mô tả bài toán
Dự án **3D Shoe Reconstruction** tập trung giải quyết bài toán tái tạo mô hình 3D hoàn chỉnh của một đôi giày chỉ từ một bức ảnh 2D duy nhất (Zero-shot Single-image 3D Reconstruction). 

Trong thực tế, việc dựng hình 3D thủ công đòi hỏi rất nhiều thời gian và kỹ năng thiết kế. Hệ thống này ứng dụng Trí tuệ Nhân tạo (kết hợp các mô hình như `Depth-Anything-V2` và `TripoSR`) để tự động hóa hoàn toàn quy trình này: từ việc tách nền ảnh, ước lượng bản đồ chiều sâu, cho đến việc đúc khối lưới đa giác (Mesh) và phủ màu (Vertex Colors) tạo thành file chuẩn `.glb` có thể tương tác trực tiếp trên giao diện Web.

Hệ thống cũng đi kèm một luồng **Human-in-the-loop** (Cơ chế thu thập phản hồi), cho phép người dùng báo cáo lỗi và hệ thống sẽ tự động tổng hợp bộ dữ liệu (Dataset) để huấn luyện lại AI trong tương lai.

---

## 📂 Cấu trúc thư mục

```text
3D-Object-Reconstruction/
│
├── data/                       # Dữ liệu vào/ra của dự án (Git bỏ qua thư mục này)
│   ├── cache/                  # Lưu mô hình 3D (GLB) đã tạo thành công để load nhanh (Caching).
│   ├── feedback/               # Lưu báo cáo lỗi (Ảnh gốc, Ảnh tách nền, GLB lỗi, Lời nhận xét).
│   ├── outputs/                # Thư mục tạm lưu kết quả của AI trong lần chạy hiện tại.
│   └── uploads/                # Ảnh 2D người dùng tải lên.
│
├── demo/
│   └── app.py                  # Mã nguồn Giao diện Web (Streamlit), xử lý UI và các nút bấm.
│
├── docs/
│   └── HUONG_DAN.md            # Tài liệu tham khảo thêm.
│
├── src/                        # Thư mục mã nguồn Cốt lõi (AI Pipeline)
│   └── pipeline.py             # File quan trọng nhất: Nhận ảnh -> Xoá phông -> TripoSR -> GLB.
│
├── tsr/                        # Mã nguồn của thuật toán lõi TripoSR (được chỉnh sửa để tối ưu).
│   └── ...
│
├── .venv/                      # Môi trường ảo chứa Python 3.11.9 và các thư viện (Không push lên Git).
├── run.bat                     # Kịch bản khởi động tự động (Kiểm tra thư viện, cài đặt, chạy Web).
├── requirements.txt            # Danh sách các thư viện cần thiết.
├── .gitignore                  # Cấu hình bỏ qua các file rác khi đẩy code lên Github.
└── README.md                   # File tài liệu bạn đang đọc.
```

---

## 🚀 Cách chạy dự án

### Cách 1: Chạy Tự động (Khuyên dùng trên Windows)
Chỉ cần mở Terminal (PowerShell hoặc CMD) và gõ:
```bash
.\run.bat
```
*Kịch bản này sẽ tự động kiểm tra xem máy đã cài đủ thư viện chưa, nếu chưa nó sẽ tự động dùng `pip` để tải về (kể cả cấu hình PyTorch cho GPU) và tự động bật Giao diện Web lên.*

### Cách 2: Chạy Thủ công
Nếu bạn muốn tự kiểm soát quá trình cài đặt:
```bash
# 1. Kích hoạt môi trường ảo
.venv\Scripts\activate

# 2. Chạy ứng dụng Streamlit
streamlit run demo/app.py
```

---

## ⚙️ Luồng xử lý (Cách Code hoạt động)

Hệ thống hoạt động theo quy trình khép kín sau:

1. **Nhận dữ liệu (app.py):** Giao diện Streamlit nhận ảnh JPG/PNG từ người dùng và băm ảnh (SHA-256) để kiểm tra xem ảnh này đã từng được xử lý và lưu trong `data/cache/` hay chưa. Nếu có, load ngay lập tức.
2. **Tiền xử lý & Xóa phông (pipeline.py):** Sử dụng thư viện `rembg` để cắt nền chiếc giày. *Đã được tinh chỉnh `alpha_matting=True` và `resize_foreground=0.80` để không lẹm mất phần đế.*
3. **Dựng hình 3D (TripoSR):** Đưa ảnh đã tách nền qua mạng Transformer của TripoSR. Lưới 3D được đúc bằng thuật toán Marching Cubes với độ phân giải cao (`resolution=384`).
4. **Hậu xử lý (pipeline.py):** Quét qua toàn bộ điểm màu (Vertex Colors) để loại bỏ dải màu Alpha (nguyên nhân gây lỗi mô hình trong suốt).
5. **Xuất file & Phản hồi (app.py):** Xuất file `.glb` hiển thị lên Web. Người dùng có thể chọn "Lưu kết quả" (vào Cache) hoặc "Báo lỗi" (Lưu thành tệp Dataset vào Feedback).

---

## 🛠 Các lỗi có thể xảy ra và Cách khắc phục

| Hiện tượng lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| **Báo lỗi "No module named '...'"** | Môi trường ảo (`.venv`) bị thiếu thư viện, hoặc bạn quên activate môi trường. | Hãy chạy file `.\run.bat`. Nó được thiết kế để tự quét và cài bù các thư viện thiếu. |
| **Cài đặt `pip` báo lỗi "Hash mismatch"** | Do file PyTorch quá nặng (2.5GB), mạng chập chờn khiến file tải bị lỗi. | File `run.bat` đã được cấu hình tải PyTorch trực tiếp từ máy chủ NVIDIA/PyTorch để tránh lỗi này. Cứ chạy lại `.\run.bat` lần nữa. |
| **Lỗi "numpy.ndarray object has no attribute ptp"** | Xung đột phiên bản Numpy cũ với Python mới (>= 3.12). | Dự án đã được thiết kế chuẩn hoá dùng **Python 3.11.9** và ép cứng `numpy < 2.0.0` trong `requirements.txt`. |
| **Mô hình 3D xuất ra bị trong suốt / Lệch màu** | Trình duyệt Web hiểu sai dải màu Alpha hoặc độ phân giải lưới 3D quá thấp. | Lỗi này đã được xử lý triệt để trong `src/pipeline.py` bằng cách nâng resolution và ép dải Alpha về 255. Nếu bạn chỉnh sửa code, hãy giữ nguyên đoạn xử lý Alpha này. |
| **Lỗi `torchmcubes` không build được C++ trên Windows** | Thư viện `torchmcubes` nguyên bản của TripoSR yêu cầu Visual Studio C++. Rất dễ gây lỗi biên dịch. | Dự án đã thay thế hoàn toàn `torchmcubes` bằng `PyMCubes` (thư viện đúc sẵn cho Windows). Đã cấu hình sẵn trong mã nguồn. |
