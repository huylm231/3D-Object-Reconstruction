# Tổng Hợp Lý Thuyết Trọng Tâm Ứng Dụng Trong Dự Án Tái Tạo 3D

Tài liệu này tổng hợp các nền tảng lý thuyết và thuật toán cốt lõi đã được nghiên cứu và ứng dụng trực tiếp vào quá trình xây dựng hệ thống tái tạo vật thể 3D (đặc biệt đối với vật thể như giày, yêu cầu giữ lại chi tiết nếp nhăn, vân vải).

---

## 1. Tiền xử lý ảnh 2D (2D Image Pre-processing)
Việc tối ưu hóa đầu vào rất quan trọng để mô hình AI có thể nhận diện và tái tạo các nếp nhăn, kết cấu của vật thể.
- **Hình thái học & Phân đoạn (Morphology & Segmentation):** Sử dụng các kỹ thuật phân đoạn (ví dụ: mask) để tìm ra Bounding Box bao sát vật thể. Cắt (Crop) bức ảnh sát viền giúp loại bỏ phông nền thừa, ép vật thể chiếm 100% diện tích, giúp các mô hình AI sau đó nội suy và tái tạo chi tiết mà không tốn tài nguyên cho phần khoảng trống.
- **Tăng cường chi tiết và Khử nhiễu:**
  - **Biến đổi Fourier (Fourier Transformation) & Lọc thông cao (High-pass filter):** Giữ lại các tần số cao trên bức ảnh, đồng nghĩa với việc duy trì độ nét của các đường viền, nếp nhăn và vân giày.
  - **Wavelet Denoising & CLAHE:** Khử nhiễu cục bộ và tăng cường độ tương phản thích ứng, giúp cải thiện chất lượng ảnh thô trước khi đưa vào các bước ước lượng chiều sâu.

---

## 2. Ước lượng chiều sâu và Trích xuất hình học (Geometry & Depth Extraction)
- **Depth Estimation (Ước lượng chiều sâu tĩnh):** Ứng dụng mạng học sâu (Neural Networks) để dự đoán bản đồ chiều sâu (depth map) trên những bức ảnh đã được Crop và làm nét. Độ chi tiết của bản đồ chiều sâu tăng đáng kể so với ảnh gốc chưa qua xử lý.
- **Structure from Motion (SfM) & Multi-View Stereo (MVS):**
  - **COLMAP / GLOMAP:** Phân tích ảnh chụp từ nhiều góc độ quanh vật thể để tính toán vị trí camera (camera pose) và mây điểm thưa (Sparse Point Cloud). Đây là dữ liệu đầu vào bắt buộc để khởi tạo các điểm cho Gaussian Splatting hoặc NeRF. GLOMAP giúp quá trình này tính toán đồng thời toàn cục (global) thay vì tuần tự (incremental), tăng tốc độ đáng kể.
  - **MVSNet:** Mạng học sâu ước lượng chiều sâu từ đặc trưng 2D, xây dựng cost volume 3D, hiệu quả cao khi xử lý ảnh thiếu chi tiết bề mặt.

---

## 3. Tái tạo không gian 3D (3D Space Reconstruction & Rendering)
- **Mô hình kiến trúc lớn (LRM) - TripoSR:** Ứng dụng Transformer để nội suy trực tiếp trường NeRF tri-plane từ một bức ảnh đơn duy nhất (Single-Image to 3D). Phù hợp tuyệt đối cho chức năng Fast-Demo của dự án (ra kết quả nhanh từ 1 bức ảnh).
- **Neural Implicit Surfaces & SDF (Signed Distance Function):**
  - Khác với NeRF truyền thống dự đoán mật độ đám mây sương (Density), các mô hình **NeuS, Geo-NeuS, HF-NeuS** dự đoán khoảng cách đến bề mặt vật thể (SDF).
  - Về mặt toán học, hàm $SDF(x)$ tại điểm $x$ trong không gian được định nghĩa như sau:
    - $SDF(x) = 0$ nếu $x$ nằm đúng trên bề mặt vật thể.
    - $SDF(x) > 0$ nếu $x$ nằm ngoài vật thể.
    - $SDF(x) < 0$ nếu $x$ nằm bên trong vật thể.
  - Định nghĩa này giúp xác định bề mặt vật thể ($SDF=0$) một cách rõ ràng, rắn chắc, hỗ trợ trực tiếp cho quá trình tạo Mesh. Biến thể HF-NeuS đặc biệt tốt trong việc tái hiện bề mặt có nhiều nếp nhăn và họa tiết nhỏ do có bù đắp từ các dải tần số cao.
- **Gaussian Splatting (3D & 2D):** 
  - Kỹ thuật tối ưu thay thế việc dò tia (Ray Marching). Rải hàng triệu khối Elip 3D (hoặc dẹt thành đĩa 2D) để rasterize thẳng lên màn hình giúp render tốc độ thời gian thực (Real-time).
  - Yêu cầu đám mây điểm thưa ban đầu từ SfM. Phiên bản 2D Gaussian Splatting cải tiến hình học, không bị "dày cộm" và dễ trích xuất ra Mesh hơn bản 3D.

---

## 4. Dựng lưới và bọc bề mặt (Mesh Reconstruction)
Chuyển đổi dữ liệu từ không gian liên tục (SDF/Tri-plane) hoặc rời rạc (Point Cloud) thành lưới bề mặt có thể sử dụng (OBJ/PLY).
- **Marching Cubes:** Thuật toán trích xuất các khối đa giác (Mesh) từ các mảng lưới thể tích có chứa thông tin SDF. Thường sử dụng độ phân giải cao (`mc_resolution=512`) trên ảnh đã Crop để tối đa hóa chi tiết lưới.
- **Screened Poisson Surface Reconstruction:** Thuật toán tái tạo bề mặt kinh điển làm mịn khối 3D từ các đám mây điểm. Việc thiết lập độ sâu phân tích lưới (`depth=10`) kết hợp tính chất "Screened" giúp bề mặt sinh ra không bị lệch hoặc biến dạng, bám sát các điểm point cloud gốc, tối ưu hơn Marching Cubes trong một số môi trường mây điểm bị nhiễu.

---

## 5. Ánh xạ vân bề mặt (UV Mapping & Texturing)
Khâu cuối cùng giúp mô hình 3D trở nên chân thực với lớp vỏ màu sắc và hoa văn.
- **UV Unwrapping (Trải phẳng UV):** Làm phẳng bề mặt mô hình 3D thành bản đồ 2D (Texture Atlas) dựa trên các đường may (seam) được cắt hợp lý. Giúp Texture được dán lại mà không bị kéo giãn hay méo mó.
- **Least Squares Conformal Maps (LSCM):** Thuật toán quasi-conformal tham số hóa bề mặt bằng cách xấp xỉ bình phương tối thiểu phương trình Cauchy-Riemann. LSCM bảo toàn hướng đa giác (không bị lật tam giác), duy trì hình dạng, giảm thiểu triệt để sự biến dạng góc (angle deformations).
  - **Điều kiện Conformal (Bảo toàn góc):** Một phép ánh xạ $X(u, v)$ được coi là conformal nếu các vector tiếp tuyến trực giao và có độ dài bằng nhau:
    $$ N(u, v) \times \frac{\partial X}{\partial u}(u, v) = \frac{\partial X}{\partial v}(u, v) $$
    Trong đó $N(u, v)$ là vector pháp tuyến của bề mặt.
  - **Phương trình Cauchy-Riemann:** Trên mặt phẳng phức ($U = u + iv$), hệ thức trên tương đương với phương trình Cauchy-Riemann:
    $$ \frac{\partial U}{\partial x} + i\frac{\partial U}{\partial y} = 0 $$
  - **Mục tiêu của LSCM:** Do trên lưới tam giác rời rạc không thể luôn thỏa mãn điều kiện vi phân này, LSCM tối thiểu hóa sai số bình phương trên toàn bộ các bề mặt tam giác $T$:
    $$ C(T) = \int_T \left| \frac{\partial U}{\partial x} + i\frac{\partial U}{\partial y} \right|^2 dA $$
  - Nhờ việc tối ưu hóa hàm mục tiêu này, thuật toán LSCM hiệu quả trong việc trải tự động các lưới phức tạp thành UV Atlas gọn gàng, giúp Texture bám chính xác vào bề mặt Mesh mà không bị lỗi biến dạng.
