# Thư mục ảnh mẫu

Đặt ảnh mẫu (giày, vật thể, ...) vào thư mục này để test các thuật toán xử lý ảnh.

## Định dạng hỗ trợ
- `.jpg`, `.jpeg`
- `.png`
- `.webp`

## Cách dùng

1. **Đặt ảnh vào đây** — ví dụ: `shoe_01.jpg`, `object_front.png`

2. **Chạy từng thuật toán riêng lẻ:**
   ```bash
   # Kích hoạt môi trường ảo
   .venv\Scripts\activate

   # Chạy module 01 - Color Spaces
   python -m image_processing.01_color_spaces

   # Chạy module 08 - Feature Detection
   python -m image_processing.08_feature_detection

   # Chạy toàn bộ pipeline
   python -m image_processing.pipeline
   ```

3. **Hoặc chỉ định ảnh cụ thể:**
   ```bash
   python -m image_processing.pipeline --image data/sample_images/shoe.jpg
   ```

4. **Kết quả** được lưu trong `data/outputs/`

## Lưu ý
- Nên dùng ảnh có **nền đơn giản** hoặc **nền trắng** để các thuật toán segmentation hoạt động tốt nhất.
- Kích thước ảnh khuyến nghị: **512×512** đến **1024×1024** pixel.
- Với bước Feature Matching (bước 9), cần **≥2 ảnh** từ các góc nhìn khác nhau.
