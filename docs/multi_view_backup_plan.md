# 🚀 Kế Hoạch Dự Phòng: Multi-View 3D Reconstruction

*Tài liệu này lưu trữ phương án nâng cấp (Backup Plan) trong trường hợp thuật toán "Single Image to 3D" (dựng 3D từ 1 bức ảnh) không đạt chất lượng hình khối như mong muốn.*

## Tình huống kích hoạt
Nếu các mô hình hiện tại (LRM / TripoSR) tạo ra mô hình 3D:
- Bị mất nét, thiếu chi tiết mặt khuất.
- Hình khối (geometry) bị méo, không phản ánh đúng hình dáng thực tế của chiếc giày.
- Không thể nội suy chính xác 360 độ từ một góc chụp duy nhất.

👉 **Chúng ta sẽ chuyển sang phương pháp Multi-View (Nhiều góc nhìn).**

## Giải pháp: COLMAP + Gaussian Splatting

Phương pháp này đòi hỏi người dùng cung cấp **1 video ngắn** quay quanh chiếc giày hoặc khoảng **20-50 bức ảnh** chụp từ nhiều góc độ khác nhau. Quy trình sẽ được nâng cấp thành 2 bước chính:

### Bước 1: Khôi phục cấu trúc không gian (Structure from Motion)
- **Công cụ:** `COLMAP` hoặc `GLOMAP` (Global Structure-from-Motion).
- **Cách thức hoạt động:** AI sẽ tìm các điểm đặc trưng chung (Feature Matching bằng SIFT/ORB) giữa các bức ảnh, từ đó tính toán chính xác vị trí đặt camera (Camera Poses) cho từng tấm ảnh và tạo ra một đám mây điểm thưa (Sparse Point Cloud).

### Bước 2: Dựng bề mặt chi tiết cao (Splatting / Implicit Surfaces)
Sau khi có được Camera Poses, chúng ta có thể dùng 1 trong 2 thuật toán "chóp bu" sau:
1. **3D/2D Gaussian Splatting:** Rải hàng triệu đốm màu (Gaussians) 3D hoặc 2D vào không gian, sau đó tối ưu hóa chúng dựa trên nhiều góc ảnh. Thuật toán này render siêu nhanh (Real-time) và đạt chất lượng hình học (geometry) siêu thực tế. Tuyệt vời nhất hiện nay.
2. **NeuS / SDFs (Neural Implicit Surfaces):** Huấn luyện một mạng Neural Network học khoảng cách bề mặt (Signed Distance Function). Thuật toán này trích xuất ra Mesh (OBJ/PLY) cực kỳ nhẵn bóng và liền mạch, dễ dàng mang đi in 3D hoặc gắn màu (texture).

## Những việc cần làm khi chuyển đổi
1. **Cài đặt thư viện C++:** Tích hợp bộ mã nguồn mở COLMAP/GLOMAP vào pipeline.
2. **Nâng cấp Giao diện (UI):** Cập nhật file `demo/app.py` để hỗ trợ tính năng upload Video hoặc chọn nhiều ảnh cùng lúc, thay vì chỉ 1 ảnh.
3. **Cấu hình phần cứng:** Gaussian Splatting cần VRAM GPU tương đối để huấn luyện. Cần thiết lập cấu hình tối ưu.

---
*Ghi chú: Tạm thời chúng ta vẫn giữ nguyên luồng 1 ảnh (Single-view) cho đến khi được kiểm chứng kỹ lưỡng.*
