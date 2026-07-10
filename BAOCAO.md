TÓM TẮT
Đề tài giải quyết bài toán tái tạo vật thể 3D từ một bức ảnh 2D duy nhất, theo hướng tiếp cận zero-shot (zero-shot single-image 3D reconstruction), nghĩa là hệ thống suy luận trực tiếp trên ảnh mới mà không cần huấn luyện lại cho từng vật thể cụ thể. Hệ thống được xây dựng như một pipeline khép kín 13 bước, tích hợp toàn bộ quy trình xử lý ảnh theo lý thuyết CVIP (Computer Vision and Image Processing) kết hợp ba nhóm mô hình học sâu: (1) phân đoạn và tách nền ảnh bằng rembg với alpha matting; (2) ước lượng bản đồ chiều sâu bằng Depth-Anything-V2 (kiến trúc Vision Transformer, encoder ViT-S); và (3) mô hình tái tạo kiến trúc lớn TripoSR (Large Reconstruction Model – LRM) dự đoán trường Neural Implicit dạng tri-plane, từ đó thuật toán Marching Cubes đúc khối lưới đa giác (mesh) ở độ phân giải 256³, tiếp theo UV Mapping bằng xatlas bake texture 2048×2048 để tô màu hoàn chỉnh.
Đầu ra của hệ thống là một mô hình 3D hoàn chỉnh có màu sắc ở định dạng .glb, có thể xoay và xem tương tác 360 độ ngay trên trình duyệt Web thông qua giao diện Streamlit. Điểm nổi bật của đề tài là tốc độ xử lý chỉ trong vài giây cho mỗi ảnh đầu vào nhờ cơ chế zero-shot và cơ chế cache dựa trên băm SHA-256; đồng thời hệ thống tích hợp cơ chế human-in-the-loop, cho phép người dùng phản hồi và báo lỗi để tích lũy dữ liệu phục vụ fine-tune mô hình trong tương lai. Đối tượng thực nghiệm chính của đề tài là giày dép — nhóm vật thể có kết cấu bề mặt (nếp nhăn, dây giày, đế giày) đòi hỏi độ chi tiết cao.
 
1. GIỚI THIỆU (INTRODUCTION)
Việc tạo ra một mô hình 3D chất lượng cao theo phương pháp thủ công đòi hỏi rất nhiều thời gian và kỹ năng dựng hình chuyên nghiệp trên các phần mềm đồ họa phức tạp như Blender, Maya hay ZBrush. Đây là rào cản lớn đối với các cá nhân, doanh nghiệp vừa và nhỏ muốn nhanh chóng số hóa sản phẩm của mình. Cùng với sự phát triển mạnh mẽ của Deep Learning trong những năm gần đây, bài toán tái tạo 3D từ ảnh (Image-based 3D Reconstruction) đã có những bước tiến vượt bậc, cho phép suy luận hình học không gian chỉ từ một hoặc một vài bức ảnh chụp thông thường.
Bài toán này có ý nghĩa ứng dụng thực tiễn rất lớn: trong thương mại điện tử, mô hình 3D giúp khách hàng xem sản phẩm ở mọi góc độ trước khi mua; trong phát triển game và nội dung số, nó rút ngắn đáng kể thời gian tạo tài nguyên đồ họa (game assets); trong lĩnh vực Thực tế ảo/Thực tế tăng cường (AR/VR), nó cho phép số hóa nhanh vật thể thật để đưa vào môi trường ảo. Đối với ngành công nghiệp thời trang và giày dép nói riêng, khả năng tái tạo chính xác các chi tiết bề mặt như nếp nhăn vải, dây giày, hoa văn đế giày là yếu tố quyết định giá trị thương mại của mô hình 3D được tạo ra.
Mục tiêu của đề tài là xây dựng một hệ thống AI tự động hóa toàn bộ quy trình biến đổi một bức ảnh 2D thành mô hình 3D hoàn chỉnh, theo hướng zero-shot (không cần huấn luyện lại cho vật thể mới), tối ưu cho trải nghiệm thời gian thực (fast-demo). Cụ thể, đề tài hướng tới các mục tiêu sau:
•	Xây dựng pipeline xử lý ảnh đầu vào tự động 13 bước theo lý thuyết CVIP: tách nền, căn chỉnh khung hình, lọc Fourier, khử nhiễu Wavelet, phát hiện đặc trưng SIFT/ORB, phân đoạn, ước lượng chiều sâu, dựng Point Cloud và Mesh 3D.
•	Ứng dụng mô hình ước lượng chiều sâu đơn ảnh (monocular depth estimation) để bổ trợ thông tin không gian và dựng Point Cloud phục vụ phân tích học thuật.
•	Ứng dụng mô hình tái tạo kiến trúc lớn (LRM) để sinh mesh 3D có màu sắc trong thời gian ngắn, xuất ra định dạng chuẩn Web (.glb).
•	Xây dựng giao diện Web tương tác, tích hợp cơ chế cache SHA-256 và cơ chế thu thập phản hồi người dùng (human-in-the-loop) làm nền tảng dữ liệu cho việc fine-tune mô hình trong các giai đoạn phát triển tiếp theo.
  
2. TỔNG QUAN LÝ THUYẾT (RELATED WORK / BACKGROUND)
2.0 Bài toán tái tạo 3D từ ảnh là gì
Tái tạo 3D từ ảnh (Image-based 3D Reconstruction) là bài toán khôi phục lại hình học không gian (geometry) và bề mặt màu sắc (texture/appearance) của một vật thể thật từ một hoặc nhiều bức ảnh 2D chụp vật thể đó. Đây là một bài toán generative/regression trong không gian 3D, khác về bản chất với các bài toán nhận dạng ảnh 2D truyền thống như Object Detection (định vị vật thể bằng bounding box) hay Semantic/Instance Segmentation (phân vùng từng pixel theo lớp): đầu ra không phải là nhãn lớp hay mặt nạ 2D, mà là một cấu trúc hình học 3D hoàn chỉnh (mesh, point cloud hoặc trường ẩn liên tục) có thể quan sát từ mọi góc độ. Bài toán được chia thành hai nhóm chính theo số lượng ảnh đầu vào: Multi-view Reconstruction (dùng nhiều ảnh) và Single-image/Zero-shot Reconstruction (chỉ dùng một ảnh duy nhất, hướng mà đề tài này lựa chọn).
2.1 Các hướng tiếp cận truyền thống: SfM và MVS
Bài toán tái tạo 3D từ ảnh (3D Reconstruction) có lịch sử phát triển lâu dài, khởi nguồn từ các phương pháp hình học kinh điển như Structure from Motion (SfM) và Multi-View Stereo (MVS). Các công cụ tiêu biểu như COLMAP hay phiên bản cải tiến GLOMAP phân tích tập hợp nhiều bức ảnh chụp từ các góc độ khác nhau quanh vật thể, thực hiện đối sánh đặc trưng (feature matching, thường dùng SIFT/ORB) để ước lượng đồng thời vị trí – hướng của camera (camera pose) và một đám mây điểm thưa (sparse point cloud). Nhược điểm cố hữu của nhóm phương pháp này là yêu cầu đầu vào nhiều ảnh (thường 20–50 ảnh hoặc một video quay quanh vật thể), không phù hợp với kịch bản người dùng chỉ có một bức ảnh duy nhất.
2.2 Các hướng tiếp cận dựa trên Deep Learning
Sự bùng nổ của Deep Learning mang lại các phương pháp biểu diễn cảnh 3D mới. Neural Radiance Fields (NeRF) biểu diễn cảnh dưới dạng một hàm liên tục ánh xạ tọa độ không gian và hướng nhìn sang mật độ (density) và màu sắc, sau đó dùng kỹ thuật dò tia (ray marching/volume rendering) để tổng hợp ảnh mới. Gaussian Splatting (3D và biến thể 2D Gaussian Splatting) thay thế việc dò tia tốn kém bằng cách rải hàng triệu khối Ellipsoid (hoặc đĩa dẹt 2D) trong không gian rồi rasterize trực tiếp, cho tốc độ render thời gian thực; tuy nhiên phương pháp này vẫn cần đám mây điểm khởi tạo từ SfM và nhiều ảnh đầu vào.
Song song đó, hướng Neural Implicit Surfaces (tiêu biểu là NeuS, Geo-NeuS, HF-NeuS) không dự đoán mật độ đám mây sương như NeRF truyền thống, mà dự đoán trực tiếp hàm khoảng cách có dấu tới bề mặt (Signed Distance Function – SDF), nhờ đó ranh giới vật thể được xác định tường minh và sắc nét hơn, thuận lợi cho việc trích xuất mesh rắn chắc ở bước sau. Biến thể HF-NeuS bổ sung dải tần số cao giúp tái hiện tốt hơn các bề mặt nhiều nếp nhăn, chi tiết nhỏ — phù hợp với đối tượng giày dép mà đề tài hướng tới.
2.3 Lựa chọn hướng tiếp cận: Large Reconstruction Model (LRM) – TripoSR
Vì mục tiêu của đề tài là tái tạo nhanh (fast-demo) chỉ từ một ảnh duy nhất theo kiểu zero-shot, các phương pháp multi-view (SfM/MVS, Gaussian Splatting cổ điển) không khả thi do yêu cầu nhiều ảnh đầu vào. Đề tài lựa chọn TripoSR — một mô hình kiến trúc lớn (Large Reconstruction Model, LRM) dựa trên Transformer — cho phép nội suy trực tiếp trường Neural Implicit dạng tri-plane từ một ảnh đơn (single-image-to-3D) chỉ trong một lượt suy luận (feed-forward), không cần tối ưu hóa lặp cho từng vật thể như NeRF/Gaussian Splatting cổ điển. Đây là lựa chọn cân bằng tốt giữa tốc độ (vài giây/ảnh) và chất lượng hình học, đáp ứng đúng yêu cầu zero-shot và trải nghiệm tức thời mà đề tài đặt ra. Mô hình ước lượng chiều sâu Depth-Anything-V2 được dùng bổ trợ để trích xuất bản đồ chiều sâu và dựng thêm point cloud minh họa phục vụ phân tích học thuật, song song với nhánh chính TripoSR.
 
3. DATASET & TIỀN XỬ LÝ DỮ LIỆU
3.1 Dữ liệu đầu vào
Vì hệ thống hoạt động theo cơ chế zero-shot, TripoSR và Depth-Anything-V2 được sử dụng ở dạng mô hình đã huấn luyện sẵn (pretrained), không đòi hỏi một bộ dataset cố định để huấn luyện lại. Dữ liệu đầu vào trong quá trình vận hành là các bức ảnh 2D bất kỳ do người dùng tải lên (định dạng PNG/JPG).
Để phục vụ kiểm thử và minh họa, đề tài xây dựng thêm một bộ ảnh mẫu (dataset_shoe) gồm 192 ảnh giày, độ phân giải 1024×1024, định dạng RGBA (đã có kênh alpha), dùng để đánh giá tính ổn định của pipeline trên nhiều kiểu dáng, màu sắc và góc chụp khác nhau. Ngoài ra, hệ thống lưu lại các đánh giá/báo lỗi từ người dùng (human-in-the-loop feedback) — gồm ảnh gốc, mô hình .glb đã sinh và mô tả lỗi dạng văn bản — trong thư mục data/feedback/, hình thành một tập dữ liệu tăng dần theo thời gian sử dụng, làm nền cho quá trình fine-tune mô hình trong tương lai.
Vì đây là bài toán generative (sinh hình học 3D) chứ không phải classification/detection, dataset không có nhãn lớp (class label) hay bounding box theo nghĩa truyền thống. Bảng dưới đây tóm tắt các đặc điểm dataset theo đúng các tiêu chí thường dùng, được diễn giải lại cho phù hợp với bài toán:
Tiêu chí	Giá trị / Diễn giải
Tên dataset	dataset_shoe (bộ ảnh mẫu tự xây dựng) + ảnh do người dùng tải lên trong quá trình vận hành thực tế
Số lượng ảnh	192 ảnh mẫu (dataset_shoe); số lượng ảnh người dùng tăng dần theo thời gian sử dụng qua data/feedback/
Số class	1 lớp đối tượng chính (giày dép); mô hình lõi (TripoSR, Depth-Anything-V2) là mô hình tổng quát, không giới hạn class cụ thể do hoạt động zero-shot
Định dạng nhãn	Không có nhãn lớp/bounding box/mask cổ điển; "nhãn" duy nhất là mô tả lỗi dạng văn bản tự do do người dùng nhập trong cơ chế human-in-the-loop (lưu kèm ảnh gốc và .glb lỗi trong data/feedback/)
3.2 Tiền xử lý (Pre-processing)
Chất lượng của bước tiền xử lý ảnh hưởng trực tiếp đến khả năng nội suy chi tiết bề mặt của mô hình AI ở các bước sau, nên được đầu tư kỹ trong pipeline:
•	Tách nền (Background Removal): sử dụng thư viện rembg với tham số alpha_matting = True, giúp giữ lại các chi tiết mảnh và phức tạp như viền, dây giày, đế giày mà không bị cắt lẹm biên dạng.
•	Căn chỉnh và thu phóng chủ thể (Foreground Resize): sau khi tách nền, hàm resize_foreground với tỉ lệ 0.85 được áp dụng để chủ thể chiếm phần lớn khung hình nhưng vẫn chừa một khoảng đệm hợp lý quanh biên, giúp mô hình 3D nhận diện đúng không gian quanh vật thể.
•	Bounding Box Cropping (tối ưu hóa): sau bước phân tích hình thái học (Bước 5), mask vật thể được dùng để xác định vùng bao (bounding box) và crop ảnh tập trung vào vật thể trước khi đưa vào TripoSR, giúp AI không lãng phí tài nguyên vào vùng nền.
•	Chuẩn hóa nền trung tính: ảnh sau khi tách nền (RGBA) được hòa (alpha-composite) với nền xám trung tính 50% thay vì nền trắng/đen thuần, giúp mô hình TripoSR không bị lệch tông màu ở các vùng biên trong suốt.
•	Tăng cường chi tiết (các bước xử lý ảnh CVIP): lọc tần số Fourier (FFT + low-pass filter), khử nhiễu Wavelet (PyWavelets, wavelet db4 level 2), phát hiện cạnh Canny/Sobel/Laplacian và cân bằng tương phản CLAHE được thực hiện trước bước ước lượng chiều sâu để cải thiện chất lượng ảnh thô.
 Data augmentation: vì TripoSR và Depth-Anything-V2 được dùng ở chế độ suy luận zero-shot (không huấn luyện lại), pipeline hiện tại không áp dụng augmentation trong lúc chạy. Tuy nhiên, để tăng độ đa dạng cho dữ liệu dùng fine-tune trong tương lai (Mục 7), đề tài đề xuất các phép augmentation phù hợp với ảnh đơn vật thể đã tách nền: xoay nhẹ (±15°), lật ngang (horizontal flip), thay đổi độ sáng/tương phản, và thay nền tổng hợp (background compositing) bằng nhiều màu nền trung tính khác nhau.
Chia train/validation/test: do hệ thống không huấn luyện mô hình từ đầu mà chỉ nạp trọng số pretrained của TripoSR và Depth-Anything-V2, khái niệm chia tập train/val/test theo nghĩa kinh điển không áp dụng cho vòng lặp hiện tại. Toàn bộ 192 ảnh trong dataset_shoe được dùng 100% cho mục đích kiểm thử (test) chất lượng pipeline. Khi triển khai hướng fine-tune trong tương lai bằng dữ liệu tích lũy từ data/feedback/, đề tài dự kiến chia theo tỉ lệ tham khảo 80% train / 10% validation / 10% test theo thời điểm thu thập, đảm bảo tập validation/test luôn chứa các mẫu mới nhất để đánh giá đúng khả năng tổng quát hoá.
 
4. PHƯƠNG PHÁP ĐỀ XUẤT (METHODOLOGY)
4.1 Kiến trúc mô hình (Backbone – Head) & Luồng xử lý Pipeline
Vì hệ thống ghép nối hai mô hình pretrained độc lập thay vì một mạng end-to-end duy nhất, khái niệm Backbone/Head được ánh xạ cho từng nhánh như sau:
Nhánh	Backbone (trích xuất đặc trưng)	Head (đầu ra chuyên biệt)
Depth-Anything-V2	Vision Transformer (ViT, encoder vits) — trích xuất đặc trưng đa tỉ lệ từ ảnh RGB	DPT decoder head — hồi quy bản đồ chiều sâu (depth map) mật độ pixel
TripoSR (LRM)	Image-to-Triplane Transformer encoder — mã hoá ảnh đơn thành biểu diễn tri-plane 3 mặt phẳng trực giao	MLP decoder dự đoán trường Neural Implicit (SDF/density + màu) tại toạ độ 3D bất kỳ, kết hợp Marching Cubes để trích xuất mesh

Toàn bộ luồng xử lý được cài đặt tập trung tại module src/image_processing/pipeline.py (pipeline chính 13 bước), kết hợp với lớp cache và giao diện tại demo/app.py. Quy trình gồm 13 bước chính, khép kín từ ảnh đầu vào tới file mô hình 3D có màu sắc:

•	Bước 1 — Chuyển đổi không gian màu (Color Spaces): ảnh đầu vào được chuyển sang Grayscale, HSV và áp dụng CLAHE (cân bằng tương phản thích ứng) để chuẩn hóa độ sáng, phục vụ các bước tiếp theo.
•	Bước 2 — Lọc tần số Fourier (Fourier Filtering): biến đổi FFT trên kênh Grayscale, áp dụng low-pass filter (radius=50) để khử nhiễu tần số cao và trích xuất phổ tần số phục vụ phân tích học thuật.
•	Bước 3 — Khử nhiễu Wavelet (Wavelet Denoising): dùng PyWavelets với wavelet db4, level=2 để phân tách ảnh thành các sub-band (LL, LH, HL, HH) và khử nhiễu bề mặt trước khi phát hiện cạnh.
•	Bước 4 — Biến đổi hình học (Geometric Transform): resize ảnh về chiều rộng cố định 400px để chuẩn hóa đầu vào cho các bước xử lý kế tiếp.
•	Bước 5 — Xử lý hình thái học (Morphology): sinh mask nhị phân của vật thể (clean mask) bằng các phép mở/đóng hình thái học, đồng thời tính gradient hình thái học để làm nổi bật biên dạng. Mask này được dùng lại ở bước Bounding Box Cropping.
•	Bước 6 — Phát hiện cạnh (Edge Detection): áp dụng đồng thời Canny (threshold 100–200), Sobel và Laplacian để trích xuất đường biên vật thể từ nhiều góc độ tần số khác nhau.
•	Bước 7 — Phân đoạn ảnh (Segmentation): phân đoạn bằng Otsu threshold và K-Means (K=4) để tách vùng màu sắc chủ đạo, hỗ trợ đánh giá vật thể phục vụ học thuật.
•	Bước 8 — Phát hiện đặc trưng (Feature Detection): trích xuất keypoints bằng SIFT và ORB trên vùng mask vật thể, lưu số lượng keypoints phục vụ so sánh độ chi tiết bề mặt.
•	Bước 9 — Đối sánh đặc trưng (Feature Matching): bước này được bỏ qua tự động vì hệ thống chỉ nhận một ảnh duy nhất (single-image mode), không có ảnh thứ hai để đối sánh.
•	[OPT] Bounding Box Cropping: dùng mask từ Bước 5 xác định bounding box vật thể (padding=20px), crop ảnh tập trung vào vật thể trước khi đưa vào TripoSR để AI không lãng phí tài nguyên vào vùng nền.
•	Bước 10 — Ước lượng chiều sâu (Depth Estimation): mô hình Depth-Anything-V2 (encoder vits, độ phân giải 518px) suy luận bản đồ chiều sâu, chuẩn hoá về [0, 1]. Bản đồ chiều sâu này được dùng ở Bước 11.
•	Bước 11 — Tạo Point Cloud: từ depth map và ảnh màu, mỗi pixel (u, v) được chiếu ngược (back-project) sang tọa độ 3D bằng mô hình camera lỗ kim, dựng Point Cloud bằng Open3D phục vụ minh hoạ học thuật.
•	Bước 12 — Dựng Mesh 3D (TripoSR): ảnh đã tách nền (foreground_ratio=0.85) được đưa vào TripoSR để dự đoán trường Neural Implicit tri-plane. Marching Cubes trích xuất mesh ở mc_resolution=256. Mesh trắng hoàn chỉnh (PBRMaterial alphaMode='OPAQUE') được xuất ra file {stem}.glb. Đồng thời callback on_mesh_ready được gọi để giao diện hiển thị mesh trắng ngay lập tức cho người dùng.
•	Bước 13 — UV Mapping & Texture: dùng xatlas để unwrap UV tự động lên toàn bộ bề mặt mesh. Sau đó query triplane của TripoSR tại nhiều góc camera khác nhau (multi-view rendering) để bake màu sắc vào UV Atlas độ phân giải 2048×2048 pixel. Xuất file {stem}_textured.glb — đây là đầu ra cuối cùng có màu sắc hoàn chỉnh.

Đầu ra thực sự của pipeline gồm 2 file GLB:
1. {stem}.glb — mesh trắng (không màu), xuất sau Bước 12, hiển thị tức thì để người dùng xem hình khối.
2. {stem}_textured.glb — mesh có màu đầy đủ (UV textured), xuất sau Bước 13, là sản phẩm cuối cùng.

 

4.2 Hàm mất mát (Loss Function)
Hệ thống sử dụng TripoSR và Depth-Anything-V2 ở chế độ suy luận zero-shot với trọng số pretrained, do đó nhóm không trực tiếp tối ưu hoá hàm mất mát trong phạm vi đề tài này. Tuy nhiên, việc hiểu rõ các hàm mất mát mà tác giả gốc dùng để huấn luyện hai mô hình lõi giúp giải thích tại sao chúng tổng quát hoá tốt trên ảnh giày mới chưa từng thấy:
•	Depth-Anything-V2 (hồi quy chiều sâu): kết hợp hàm mất mát bất biến affine/scale-shift-invariant (SSI loss) — cho phép so sánh chiều sâu dự đoán và chiều sâu tham chiếu bất kể sai lệch tỉ lệ/độ dời tuyệt đối — với một thành phần chưng cất tri thức (distillation loss) từ nhãn giả (pseudo-label) chất lượng cao sinh bởi mô hình giáo viên trên tập dữ liệu không nhãn quy mô lớn, cùng thành phần mất mát khớp gradient (gradient-matching loss) giúp giữ sắc nét các cạnh biên.
•	TripoSR / LRM (tái tạo hình học): được huấn luyện bằng hàm mất mát render đa góc nhìn (multi-view rendering loss), so sánh ảnh tổng hợp từ trường ẩn dự đoán với ảnh thật ở nhiều góc camera khác nhau, thường gồm thành phần sai số photometric (L1/L2 hoặc LPIPS perceptual loss) cộng với thành phần mất mát mặt nạ/silhouette (mask IoU loss) để ràng buộc đúng hình bao vật thể, có thể kèm số hạng chính quy hoá Eikonal (Eikonal regularization) nhằm giữ tính chất |∇SDF| ≈ 1 của trường khoảng cách.
Ltotal = Lphoto + λmask · Lmask + λeik · Leikonal   (0)
trong đó λ_mask, λ_eik là các trọng số cân bằng giữa các thành phần mất mát, được thiết lập bởi tác giả gốc của TripoSR trong quá trình huấn luyện trên tập dữ liệu 3D quy mô lớn (ví dụ Objaverse). Vì đề tài không huấn luyện lại các mô hình này, các giá trị λ cụ thể không thuộc phạm vi kiểm soát của hệ thống hiện tại.
4.3 Thiết lập huấn luyện / suy luận (Training & Inference Setup)
Không áp dụng huấn luyện lại (no training) đối với hai mô hình lõi TripoSR và Depth-Anything-V2 trong phạm vi đề tài — cả hai được nạp trực tiếp từ trọng số pretrained công khai và chạy ở chế độ suy luận (inference-only, torch.no_grad()). Do đó các khái niệm Optimizer, Learning rate, Epochs theo nghĩa huấn luyện không phát sinh ở vòng lặp hiện tại của hệ thống. Thay vào đó, bảng dưới đây liệt kê cấu hình suy luận (inference configuration) thực tế được sử dụng:
Thành phần	Cấu hình / Giá trị
Batch size (suy luận)	1 ảnh / lượt xử lý (single-image inference)
Optimizer / Learning rate / Epochs	Không áp dụng — mô hình chạy ở chế độ pretrained, inference-only
Thiết bị suy luận	GPU NVIDIA (CUDA) nếu có, tự động fallback CPU
Độ phân giải Depth-Anything-V2	518 px (encoder vits)
Chunk size renderer TripoSR	8192
Độ phân giải Marching Cubes	256³ (mc_resolution = 256)
Foreground ratio (resize_foreground)	0.85
Độ phân giải UV Texture	2048×2048 px (texture_res = 2048)
Đề xuất cho tương lai: nếu triển khai fine-tune trên dữ liệu tích luỹ từ data/feedback/ (Mục 7), nhóm dự kiến sử dụng optimizer AdamW, learning rate khởi tạo khoảng 1e-4 với lịch giảm dần (cosine decay), huấn luyện theo số epoch nhỏ (fine-tune, ví dụ 10–20 epoch) trên batch size 4–8 tuỳ dung lượng GPU, đồng thời theo dõi đường cong loss trên tập validation để tránh overfitting trên tập dữ liệu phản hồi còn hạn chế.
4.4 Cơ chế Cache
Hệ thống cài đặt cơ chế cache chính xác theo mã băm SHA-256: ảnh đầu vào được băm SHA-256 và mã băm dùng làm khoá tra cứu trong thư mục data/cache/. Nếu ảnh đã từng được xử lý, file .glb tương ứng được nạp lại tức thời mà không cần chạy lại toàn bộ pipeline AI.

Lưu ý về cache gần đúng HSV: hệ thống đã thiết kế và cài đặt cơ chế so khớp ảnh gần giống (near-duplicate matching) dựa trên histogram màu trong không gian HSV, với công thức tương quan (correlation) của OpenCV:

d(H1, H2) =  ΣI (H1(I) − H̄1)(H2(I) − H̄2)  ⁄  √[ ΣI(H1(I) − H̄1)² · ΣI(H2(I) − H̄2)² ]   (1)

Ảnh được resize về 256×256, chuyển sang HSV và tính histogram 2 kênh (Hue, Saturation) với 50×60 bin. Tuy nhiên, tính năng này đã bị tạm thời vô hiệu hóa trong phiên bản hiện tại (hàm find_similar_cached_model trả về None, 0.0) do gây ra false positive nghiêm trọng: hai ảnh giày khác nhau nhưng có cùng phông nền trắng đều cho hệ số tương quan cao, dẫn đến trả nhầm mô hình 3D. Hệ thống hiện tại chỉ sử dụng exact hash matching (SHA-256). Cơ chế cache HSV sẽ được cải thiện bằng cách kết hợp thêm đặc trưng hình dạng (shape embedding) ở các phiên bản tiếp theo.
4.5 Cơ sở toán học (Theoretical Formulas)
4.5.1 Mô hình camera lỗ kim và dựng đám mây điểm từ Depth Map
Từ bản đồ chiều sâu chuẩn hoá và ảnh màu tương ứng, mỗi điểm ảnh (u, v) với giá trị chiều sâu Z(u, v) được chiếu ngược (back-project) thành một điểm 3D theo mô hình camera lỗ kim (pinhole camera model):
X = (u − cx) · Z / fx ,     Y = (v − cy) · Z / fy ,     Z = Z(u, v)   (2)
trong đó tiêu cự và tâm ảnh được xấp xỉ theo kích thước ảnh đầu vào: fₓ = f_y = max(H, W), cₓ = W⁄2, c_y = H⁄2, với H, W lần lượt là chiều cao và chiều rộng ảnh. Đám mây điểm 3D thu được (kèm pháp tuyến ước lượng bằng Open3D) chủ yếu phục vụ minh hoạ và đối chiếu học thuật, không phải là nhánh tạo mesh chính thức của hệ thống.
4.5.2 Neural Implicit Surfaces & Signed Distance Function (SDF)
SDF xác định khoảng cách có dấu từ một điểm bất kỳ trong không gian đến bề mặt vật thể gần nhất. Về mặt toán học, hàm SDF(x) tại điểm x được định nghĩa:
SDF(x) = 0   nếu x nằm đúng trên bề mặt vật thể
SDF(x) > 0   nếu x nằm ngoài vật thể
SDF(x) < 0   nếu x nằm bên trong vật thể   (3)
Ranh giới vật thể được xác định tường minh tại tập hợp {x | SDF(x) = 0}. Chính tính chất này giúp thuật toán Marching Cubes trích xuất mesh rắn chắc, sắc nét — trong hệ thống, trường ẩn này được TripoSR dự đoán gián tiếp thông qua biểu diễn tri-plane, và Marching Cubes được thực thi ở độ phân giải lưới 256³ để cân bằng giữa chất lượng chi tiết bề mặt và thời gian xử lý.
4.5.3 UV Mapping và Ánh xạ vân bề mặt
Sau khi có mesh (Bước 12), hệ thống thực hiện UV Mapping (Bước 13) bằng thư viện xatlas — một thư viện C++ binding sử dụng các thuật toán conformal parameterization để unwrap tự động bề mặt 3D thành UV Atlas 2D mà không gây biến dạng góc. Về mặt lý thuyết, thuật toán xấp xỉ bài toán Least Squares Conformal Maps (LSCM): tối thiểu hóa sai số bình phương của phương trình Cauchy–Riemann trên lưới tam giác rời rạc:
C(T) = ∫T  | ∂U⁄∂x + i ∂U⁄∂y |2  dA   (4)
Sau khi có UV Atlas, hệ thống query triplane của TripoSR tại nhiều góc camera (multi-view rendering) để lấy màu sắc bề mặt, bake vào ảnh texture UV 2048×2048, và xuất file .glb có material PBR đầy đủ (alphaMode='OPAQUE').
4.6 Bảng tổng hợp cấu hình hệ thống (Implementation Setup)
Thành phần	Cấu hình / Giá trị
Ngôn ngữ & framework	Python 3.11.x, PyTorch, Streamlit
Xử lý ảnh CVIP	OpenCV, PyWavelets, scikit-image
Tách nền	rembg, alpha_matting = True
Thu phóng chủ thể	resize_foreground = 0.85
Depth estimation	Depth-Anything-V2, encoder vits, độ phân giải suy luận 518 px
Mô hình 3D	TripoSR (stabilityai/TripoSR), chunk size renderer = 8192
Trích xuất mesh	Marching Cubes, mc_resolution = 256
Hậu xử lý mesh	PBRMaterial(alphaMode='OPAQUE'), fix_normals(), repair topology
UV Mapping	xatlas (conformal unwrap tự động)
Texture baking	Triplane multi-view query, texture_res = 2048×2048
Cache chính xác	Băm SHA-256 theo nội dung ảnh
Cache gần đúng	Đã thiết kế (Histogram HSV, 50×60 bin, ngưỡng 0.95) nhưng tạm thời TẮT do false positive
Thiết bị suy luận	GPU NVIDIA (CUDA) hoặc CPU (fallback tự động)

 
5. THỰC NGHIỆM & KẾT QUẢ
5.1 Metric đánh giá
Vì đầu ra của hệ thống là một mesh 3D chứ không phải nhãn lớp hay mặt nạ phân đoạn 2D, các chỉ số kinh điển của Detection (mAP, Precision, Recall) hay Segmentation (IoU, Dice, Pixel Accuracy) không áp dụng trực tiếp. Thay vào đó, đề tài sử dụng các chỉ số tiêu chuẩn của lĩnh vực 3D Reconstruction, cùng một số chỉ số vận hành (operational metrics) đặc thù của hệ thống:
•	Chamfer Distance (CD): khoảng cách trung bình hai chiều giữa tập điểm lấy mẫu trên mesh dự đoán P và mesh/point cloud tham chiếu Q (nếu có dữ liệu 3D ground-truth, ví dụ quét từ máy scan 3D):
CD(P, Q) = (1⁄|P|) Σp∈P minq∈Q ‖p − q‖2  +  (1⁄|Q|) Σq∈Q minp∈P ‖p − q‖2   (5)
•	Volumetric IoU: tỉ lệ giao trên hợp giữa khối (voxel) chiếm bởi mesh dự đoán và mesh tham chiếu, đóng vai trò tương tự IoU trong segmentation nhưng ở không gian 3D.
•	F-score@τ: tỉ lệ điểm trên bề mặt dự đoán nằm trong ngưỡng khoảng cách τ so với bề mặt thật (và ngược lại), kết hợp giữa precision và recall hình học.
•	Normal Consistency (NC): độ tương đồng trung bình giữa vector pháp tuyến của bề mặt dự đoán và bề mặt tham chiếu tại các điểm gần nhau, phản ánh độ mượt/độ đúng hướng của mesh.
•	Chỉ số vận hành bổ sung (đặc thù hệ thống, đo được trực tiếp không cần ground-truth 3D): thời gian xử lý trung bình mỗi ảnh (13 bước + UV mapping), tỉ lệ cache-hit (SHA-256 exact match), và tỉ lệ phản hồi lỗi từ người dùng (số lượt báo lỗi ⁄ tổng số lượt sử dụng) thu thập qua cơ chế human-in-the-loop.
Lưu ý: các chỉ số CD/IoU/F-score/NC ở trên đòi hỏi một tập mesh 3D tham chiếu (ground-truth) để so sánh — hệ thống hiện tại chưa có bộ ground-truth 3D tương ứng với dataset_shoe, do đó các kết quả định lượng ở Mục 5.3 tạm thời tập trung vào các chỉ số vận hành đo được trực tiếp; nhóm có thể bổ sung ground-truth (ví dụ quét 3D một số mẫu giày thật) để tính đầy đủ các chỉ số hình học này trong các phiên bản báo cáo tiếp theo.
5.2 Thiết lập thực nghiệm
Hệ thống được thiết kế để chạy trên phần cứng đa dạng. Trên máy có GPU NVIDIA, PyTorch với CUDA được sử dụng để tăng tốc cả hai nhánh Depth-Anything-V2 và TripoSR. Trên máy không có GPU rời, hệ thống tự động chuyển sang chạy trên CPU, đồng thời sử dụng cấu hình encoder nhẹ nhất (vits) của Depth-Anything-V2 để đảm bảo thời gian phản hồi vẫn ở mức chấp nhận được cho một buổi demo trực tiếp (fast-demo). Thực nghiệm được tiến hành trên bộ ảnh mẫu dataset_shoe (192 ảnh giày, 1024×1024) và các ảnh giày do người dùng tự chụp, đa dạng về góc chụp, ánh sáng và kiểu dáng.
5.3 Kết quả định lượng và hiệu năng
Với một ảnh chưa từng xử lý, hệ thống thực hiện đầy đủ 13 bước xử lý ảnh CVIP + TripoSR + UV Mapping, xuất ra 2 file GLB (mesh trắng và mesh có màu). Giao diện hiển thị mesh trắng ngay sau Bước 12, sau đó cập nhật sang mesh có màu sau khi Bước 13 hoàn thành, tạo cảm giác phản hồi tức thì cho người dùng. Với ảnh trùng khớp chính xác (theo băm SHA-256), hệ thống trả kết quả gần như tức thời do bỏ qua hoàn toàn bước suy luận AI.

Bảng dưới đây so sánh pipeline cũ (src/pipeline.py) với pipeline mới 13 bước (src/image_processing/pipeline.py) đang được sử dụng:
Tiêu chí	Pipeline cũ (src/pipeline.py)	Pipeline mới 13 bước (src/image_processing/pipeline.py)
Số bước xử lý	2 bước (Depth + TripoSR)	13 bước (toàn bộ CVIP + AI)
foreground_ratio	0.80	0.85
mc_resolution (Marching Cubes)	384³	256³
Xử lý Alpha	Ép vertex_color[:, 3] = 255	PBRMaterial(alphaMode='OPAQUE')
Texture	Vertex Colors (không UV)	UV Atlas 2048×2048 (xatlas + triplane bake)
Cache gần đúng	SHA-256 + HSV (ngưỡng 0.95)	SHA-256 only (HSV đã tắt do false positive)
Bounding Box Crop	Không có	Có (tự động từ mask Bước 5)
Đầu ra	1 file .glb	2 file: {stem}.glb (trắng) + {stem}_textured.glb (có màu)
Thời gian xử lý / ảnh	[Điền số đo — GPU/CPU]	[Điền số đo — GPU/CPU]

 

5.4 Kết quả định tính
Về mặt định tính, hệ thống tái tạo tốt hình khối tổng thể của giày, giữ được các đặc trưng không gian như phần rỗng cổ giày, độ cong của đế và một phần nếp gấp trên bề mặt vải/da, nhờ hiệu quả của bước tách nền với alpha matting kết hợp Bounding Box Cropping tự động. Mesh trắng (Bước 12) thể hiện hình học chính xác và sắc nét. Mesh có màu (Bước 13, UV textured) tái hiện màu sắc bề mặt đồng đều hơn so với vertex colors, do texture baking từ nhiều góc camera hạn chế được hiện tượng seam và màu không đều. Bản đồ chiều sâu do Depth-Anything-V2 sinh ra thể hiện rõ ràng tương phản giữa các vùng gần (mũi giày) và xa (gót giày) trong ảnh gốc, phù hợp trực quan với hình dạng thật của vật thể.

 


 


 

6. PHÂN TÍCH & THẢO LUẬN
Ưu điểm
•	Tốc độ nhanh, phù hợp ứng dụng thời gian thực/fast-demo: chỉ cần một ảnh duy nhất và một lượt suy luận feed-forward, không cần tối ưu hoá lặp như NeRF/Gaussian Splatting cổ điển.
•	Pipeline 13 bước toàn diện: tích hợp đầy đủ các kỹ thuật xử lý ảnh CVIP (Fourier, Wavelet, Morphology, Feature Detection, Segmentation,...) trước khi đưa vào AI, đảm bảo tính học thuật và khả năng phân tích từng giai đoạn xử lý.
•	Đơn giản hoá trải nghiệm người dùng: không cần máy quét 3D chuyên dụng hay chụp nhiều góc ảnh như các phương pháp SfM/MVS.
•	Cơ chế cache SHA-256 giúp tiết kiệm tài nguyên GPU/CPU đáng kể cho các lượt sử dụng lặp lại (cùng ảnh).
•	Đầu ra 2 file GLB (mesh trắng tức thì + mesh textured cuối cùng) tạo trải nghiệm phản hồi tiến dần (progressive feedback) cho người dùng.
•	Cơ chế human-in-the-loop tạo tiền đề thu thập dữ liệu thực tế có gán nhãn lỗi, phục vụ fine-tune mô hình về sau mà không cần một bộ dataset gán nhãn thủ công tốn kém ban đầu.
Hạn chế & Phân tích lỗi
Vì hệ thống không phải là bộ phân loại/phát hiện vật thể, khái niệm False Positive (FP) / False Negative (FN) được diễn giải lại theo ngữ cảnh của các cơ chế có tính "quyết định nhị phân" trong pipeline — lớp cache và bước tách nền:
•	False Positive của cache SHA-256: không xảy ra (SHA-256 là exact match, xác suất đụng độ hash cực kỳ thấp).
•	False Positive của cache HSV (đã tắt): hệ thống có thể nhận nhầm hai ảnh khác vật thể là "giống nhau" khi có cùng phông nền trắng hoặc cùng tông màu chủ đạo. Đây là lý do tính năng này bị tắt trong phiên bản hiện tại.
•	Lỗi tách nền (nguồn sai số cho bước dựng mesh): rembg tách sót một phần nền (false negative của mask nền) hoặc cắt lẹm vào vật thể thật (false positive của mask nền) đều lan truyền trực tiếp thành sai số hình học ở mesh cuối cùng, do TripoSR không có cơ chế tự sửa lỗi mặt nạ đầu vào.
•	Góc khuất (Occlusions): vì chỉ dùng một ảnh duy nhất, các bề mặt hoàn toàn khuất phía sau vật thể phải được mô hình Transformer suy đoán (hallucination) dựa trên tri thức đã học từ dữ liệu huấn luyện, đôi khi dẫn đến hoạ tiết mặt sau không khớp hoàn toàn với vật thể thật.
•	Chi tiết mỏng và cấu trúc dạng lưới đan: các chi tiết rất mảnh (dây giày mảnh, lưới thoáng khí) đôi khi bị làm mượt hoặc mất do giới hạn độ phân giải không gian của lưới Marching Cubes (256³); tăng độ phân giải sẽ cải thiện chi tiết nhưng đánh đổi thời gian xử lý và bộ nhớ.
•	Chất lượng đầu ra phụ thuộc nhiều vào chất lượng tách nền ở bước tiền xử lý; nếu rembg tách nền sai (lẹm biên hoặc sót nền), sai số này sẽ lan truyền và ảnh hưởng trực tiếp đến hình học mesh cuối cùng.
 
7. KẾT LUẬN & HƯỚNG PHÁT TRIỂN
Kết luận
Đề tài đã xây dựng thành công một quy trình khép kín 13 bước, tự động biến đổi một bức ảnh 2D thành mô hình 3D có màu sắc ở định dạng GLB, tích hợp toàn bộ kỹ thuật xử lý ảnh CVIP (Color Spaces, Fourier Filtering, Wavelet Denoising, Morphology, Edge Detection, Segmentation, Feature Detection) trước khi ứng dụng các công nghệ AI hiện đại (tách nền có alpha matting, ước lượng chiều sâu bằng Vision Transformer, tái tạo hình học bằng Large Reconstruction Model kết hợp Marching Cubes, UV Mapping & Texture baking bằng xatlas). Hệ thống khắc phục được các vấn đề kỹ thuật thực tế như lẹm nền, lỗi hiển thị vật liệu trong suốt trên môi trường Web 3D (thông qua PBRMaterial OPAQUE), đồng thời tối ưu hiệu năng bằng cơ chế cache SHA-256, đáp ứng tốt mục tiêu zero-shot và trải nghiệm gần như tức thời đã đề ra.
Bài học rút ra
•	Pipeline xử lý ảnh CVIP đầy đủ (13 bước) mang lại giá trị học thuật rõ ràng và khả năng debug từng giai đoạn, dù các bước đầu (Fourier, Wavelet, Feature Detection) không trực tiếp ảnh hưởng đến chất lượng mesh cuối — giá trị chính nằm ở Bounding Box Cropping (tự động từ mask Bước 5), Depth Estimation (Bước 10) và UV Mapping (Bước 13).
•	Chất lượng tiền xử lý (tách nền, Bounding Box Crop, chuẩn hoá khung hình) ảnh hưởng đến kết quả cuối cùng nhiều hơn cả việc chọn mô hình 3D lõi.
•	Các lỗi kỹ thuật tưởng như "vặt" (ví dụ kênh Alpha của material, cách export GLB) có thể làm hỏng hoàn toàn trải nghiệm hiển thị trên môi trường Web 3D dù phần hình học bên dưới hoàn toàn đúng — kiểm thử trực quan trên trình duyệt thực tế là bước không thể bỏ qua.
•	Cache gần đúng dựa trên histogram màu HSV có nguy cơ false positive cao khi ảnh có phông nền trắng đồng nhất; cần kết hợp thêm đặc trưng hình dạng để đạt độ tin cậy thực tế.
Hướng phát triển
•	Tích hợp phương án dự phòng Multi-view (COLMAP/GLOMAP kết hợp 3D/2D Gaussian Splatting hoặc NeuS) cho các trường hợp người dùng có thể cung cấp một video ngắn hoặc 20–50 ảnh chụp quanh vật thể, nhằm đạt độ chính xác hình học 360 độ cao hơn khi phương án single-image không đáp ứng đủ chất lượng mong muốn.
•	Tận dụng dữ liệu phản hồi tích luỹ trong data/feedback/ (ảnh, mô hình lỗi, mô tả lỗi) để fine-tune lại phần đầu trích xuất đặc trưng hoặc đầu dự đoán trường ẩn của TripoSR cho các lớp vật thể chuyên biệt (đặc biệt là giày dép).
•	Khôi phục và cải thiện cơ chế cache gần đúng bằng cách kết hợp thêm đặc trưng hình dạng (shape embedding) bên cạnh histogram màu HSV, nhằm giảm rủi ro false positive giữa các vật thể có màu sắc tương đồng nhưng kiểu dáng khác nhau.
•	Nâng cấp độ phân giải Marching Cubes từ 256³ lên 384³ hoặc cao hơn khi có GPU đủ mạnh, để cải thiện khả năng tái hiện nếp nhăn và hoạ tiết bề mặt mảnh (dây giày, lưới thoáng khí).
 
8. PHỤ LỤC (APPENDIX)
Cấu trúc dự án
Đường dẫn	Vai trò
src/image_processing/pipeline.py	Pipeline chính 13 bước: Color Spaces → Fourier → Wavelet → Geometric → Morphology → Edge → Segmentation → Feature Detection → Feature Matching → Depth → Point Cloud → Mesh TripoSR → UV Mapping & Texture.
src/pipeline.py	Pipeline đơn giản (cũ, chỉ dùng làm fallback): Depth estimation + TripoSR, không có 13 bước CVIP.
src/image_processing/01_color_spaces.py	Bước 1: Chuyển đổi không gian màu (Grayscale, HSV, CLAHE).
src/image_processing/02_fourier_filtering.py	Bước 2: Lọc tần số Fourier (FFT, low-pass filter, radius=50).
src/image_processing/03_wavelet_denoising.py	Bước 3: Khử nhiễu Wavelet (PyWavelets, wavelet=db4, level=2).
src/image_processing/04_geometric_transform.py	Bước 4: Biến đổi hình học (resize về 400px).
src/image_processing/05_morphology.py	Bước 5: Hình thái học + sinh clean mask vật thể.
src/image_processing/06_edge_detection.py	Bước 6: Phát hiện cạnh (Canny threshold 100–200, Sobel, Laplacian).
src/image_processing/07_segmentation.py	Bước 7: Phân đoạn ảnh (Otsu threshold, K-Means K=4).
src/image_processing/08_feature_detection.py	Bước 8: Phát hiện đặc trưng (SIFT, ORB).
src/image_processing/09_feature_matching.py	Bước 9: Đối sánh đặc trưng (bỏ qua tự động vì single-image).
src/image_processing/10_depth_estimation.py	Bước 10: Ước lượng chiều sâu (Depth-Anything-V2, encoder vits, 518px).
src/image_processing/11_point_cloud.py	Bước 11: Tạo Point Cloud từ depth map (Open3D, pinhole back-projection).
src/image_processing/12_mesh_reconstruction.py	Bước 12: Dựng Mesh 3D (TripoSR, mc_resolution=256, foreground_ratio=0.85, PBRMaterial OPAQUE). Xuất {stem}.glb trắng.
src/image_processing/13_uv_mapping.py	Bước 13: UV Mapping & Texture (xatlas unwrap, triplane multi-view bake, texture_res=2048). Xuất {stem}_textured.glb có màu.
src/depth_anything_v2/	Mã nguồn mô hình ước lượng chiều sâu Depth-Anything-V2.
src/tsr/	Mã nguồn lõi của mô hình TripoSR (TSR system, tiện ích xử lý ảnh nền).
src/models/	Các wrapper kết nối tới các mô hình AI.
demo/app.py	Giao diện Web Streamlit: tải ảnh, hiển thị mesh trắng tức thì (Bước 12) rồi mesh textured (Bước 13), cache SHA-256, thu thập phản hồi người dùng.
docs/ly_thuyet_ap_dung.md	Tài liệu tổng hợp lý thuyết và công thức toán học/AI đã ứng dụng.
docs/multi_view_backup_plan.md	Kế hoạch dự phòng Multi-view (COLMAP/GLOMAP + Gaussian Splatting/NeuS).
dataset/dataset_shoe/	Bộ ảnh mẫu 192 ảnh giày (1024×1024, RGBA) dùng để kiểm thử pipeline.
data/cache/	Lưu trữ mô hình .glb đã tạo thành công theo mã băm SHA-256.
data/feedback/	Lưu báo lỗi của người dùng (ảnh, .glb, mô tả lỗi) — dữ liệu cho fine-tune tương lai.
weights/depth_anything_v2_vits.pth	Trọng số (checkpoint) của mô hình Depth-Anything-V2, encoder vits.
Thông số & biểu đồ huấn luyện chi tiết
Bảng thông số suy luận chi tiết (batch size, độ phân giải, thiết bị,...) đã được trình bày tại Mục 4.3 và 4.6. Vì hai mô hình lõi TripoSR và Depth-Anything-V2 được sử dụng ở chế độ pretrained/zero-shot và không được huấn luyện lại trong phạm vi đề tài, hệ thống hiện chưa có biểu đồ đường cong huấn luyện (loss curve, learning-rate schedule) của riêng nhóm. Khi triển khai hướng fine-tune trên dữ liệu feedback trong tương lai (Mục 7), nhóm dự kiến bổ sung tại đây các biểu đồ loss theo epoch trên tập train/validation, cùng đường cong các chỉ số hình học (Chamfer Distance, IoU thể tích) theo Mục 5.1 để theo dõi quá trình cải thiện mô hình.
Link mã nguồn dự án: https://github.com/huylm231/3D-Object-Reconstruction
 
TÀI LIỆU THAM KHẢO
•	TripoSR: Fast 3D Object Reconstruction from a Single Image — Stability AI & Tripo AI, https://github.com/VAST-AI-Research/TripoSR
•	Depth Anything V2 — Yang, L. et al., https://github.com/DepthAnything/Depth-Anything-V2
•	NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction — Wang, P. et al.
•	Geo-NeuS và HF-NeuS: các biến thể cải tiến của NeuS cho tái tạo bề mặt Neural Implicit.
•	Marching Cubes: A High Resolution 3D Surface Construction Algorithm — Lorensen, W. E. & Cline, H. E.
•	Least Squares Conformal Maps for Automatic Texture Atlas Generation — Lévy, B. et al.
•	xatlas — UV unwrapping library, https://github.com/jpcy/xatlas
•	3D Gaussian Splatting for Real-Time Radiance Field Rendering — Kerbl, B. et al.
•	Structure-from-Motion Revisited (COLMAP) — Schönberger, J. L. & Frahm, J.-M.
•	rembg — thư viện tách nền ảnh mã nguồn mở, https://github.com/danielgatis/rembg
•	Open3D: A Modern Library for 3D Data Processing — Zhou, Q.-Y. et al.
•	PyWavelets — Wavelet transforms in Python, https://github.com/PyWavelets/pywt
•	Streamlit — framework xây dựng giao diện Web cho ứng dụng dữ liệu/AI, https://streamlit.io
•	Mã nguồn dự án: https://github.com/huylm231/3D-Object-Reconstruction
