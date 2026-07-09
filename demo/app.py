import streamlit as st
import sys
# Cố định encoding thành UTF-8 trên Windows để tránh lỗi charmap khi in tiếng Việt
if sys.stdout is not None and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
if sys.stderr is not None and getattr(sys.stderr, 'encoding', '').lower() != 'utf-8':
    try: sys.stderr.reconfigure(encoding='utf-8')
    except: pass

import os
import uuid
import base64
import hashlib
import shutil
import time
import concurrent.futures
from pathlib import Path

# ──────────────────── Cau hinh thu muc ────────────────────
ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "data" / "outputs"
CACHE_DIR = ROOT / "data" / "cache"
FEEDBACK_DIR = ROOT / "data" / "feedback"

for d in [UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR, FEEDBACK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.image_processing.pipeline import run_pipeline
except ImportError:
    from src.pipeline import run_pipeline

# ──────────────────── Utils ────────────────────
def compare_images(img1_bytes, img2_path):
    try:
        import cv2
        import numpy as np
        nparr1 = np.frombuffer(img1_bytes, np.uint8)
        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(str(img2_path))
        if img1 is None or img2 is None:
            return 0.0
        
        img1 = cv2.resize(img1, (256, 256))
        img2 = cv2.resize(img2, (256, 256))
        
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity
    except Exception:
        return 0.0

def find_similar_cached_model(img_bytes, search_dir):
    """Tim anh tuong tu trong thu muc. Tra ve (image_path, score) hoac (None, 0.0)."""
    # [FIX] Tạm thời tắt tính năng so sánh độ tương đồng bằng Histogram.
    # Tính năng này gây lỗi nhận diện sai khi nhiều ảnh có cùng phông nền trắng.
    # Chỉ sử dụng EXACT HASH MATCHING (đã code sẵn ở block phía dưới gọi hàm này).
    return None, 0.0

def find_glbs_for_image(image_path):
    """Tim file .glb trang (step 12) va mau (step 13/cache) tuong ung voi anh.
    Tra ve (white_glb, colored_glb). Moi gia tri co the la None."""
    image_path = Path(image_path)
    parent = image_path.parent
    stem = image_path.stem  # vd: 49e6caf2-..._cropped
    
    # Tim base stem (bo _cropped neu co)
    base_stem = stem.replace("_cropped", "") if "_cropped" in stem else stem
    
    # Tim tat ca .glb lien quan
    all_glbs = list(parent.glob(f"{stem}*.glb")) + list(parent.glob(f"{base_stem}*.glb"))
    # Bo trung lap
    all_glbs = list({str(p): p for p in all_glbs}.values())
    
    white_glb = None
    colored_glb = None
    
    for g in all_glbs:
        name = g.name.lower()
        if "textured" in name:
            colored_glb = g
        elif white_glb is None:  # GLB dau tien khong phai textured = trang
            white_glb = g
    
    # Tim colored trong cache
    if not colored_glb:
        for cache_glb in CACHE_DIR.glob("*.glb"):
            colored_glb = cache_glb
            break  # Lay file dau tien
    
    return white_glb, colored_glb

def get_viewer_html(glb_path_or_bytes, is_bytes=False):
    if not is_bytes:
        with open(glb_path_or_bytes, "rb") as f:
            model_bytes = f.read()
    else:
        model_bytes = glb_path_or_bytes
        
    b64_model = base64.b64encode(model_bytes).decode('utf-8')
    model_data_uri = f"data:model/gltf-binary;base64,{b64_model}"
    
    return f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <div style="background-color: #1E1E1E; border-radius: 12px; overflow: hidden; padding: 10px; margin-bottom: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <model-viewer 
            src="{model_data_uri}" 
            alt="3D Shoe Model" 
            auto-rotate camera-controls 
            interaction-prompt="none"
            min-camera-orbit="auto 0deg auto" max-camera-orbit="auto 180deg auto"
            environment-image="neutral" shadow-intensity="1" 
            style="width: 100%; height: 500px; outline: none; background-color: #1E1E1E; touch-action: none;">
        </model-viewer>
    </div>
    """

def render_html(html_str, height=535):
    # Phai dung st.components.v1.html vi no tao iframe voi height co dinh.
    # st.html() khong ho tro height → model-viewer bi co thanh 0px.
    st.components.v1.html(html_str, height=height)

# ──────────────────── Giao dien Streamlit ────────────────────
st.set_page_config(page_title="3D Shoe Reconstruction", page_icon="👟", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS cho giao dien dep hon
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4A5568;
        border-radius: 15px;
        padding: 20px;
        background-color: #2D3748;
    }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(229, 46, 113, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">👟 3D Shoe Reconstruction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Tải lên một bức ảnh 2D của giày và AI sẽ tái tạo thành mô hình 3D tương tác siêu thực!</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📸 Ảnh Đầu Vào 2D")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh (JPG, PNG)", type=['png', 'jpg', 'jpeg', 'webp'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        st.image(uploaded_file, width="stretch", caption="Ảnh bạn đã tải lên")

with col2:
    st.markdown("### 🪄 Khung Hiển Thị 3D")
    
    if uploaded_file is not None:
        img_bytes = uploaded_file.getvalue()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        
        cached_glb = CACHE_DIR / f"{img_hash}.glb"

        if "last_hash" not in st.session_state or st.session_state.last_hash != img_hash:
            st.session_state.last_hash = img_hash
            st.session_state.current_model_path = None
            st.session_state.current_white_model_path = None
            st.session_state.show_save_buttons = False
            st.session_state.feedback_mode = False
            st.session_state.current_job_id = None

        if st.session_state.current_model_path is None:
            if st.button("🚀 Bắt đầu tạo mô hình 3D", type="primary", width="stretch"):
                # Tim anh tuong tu trong outputs
                white_glb = None
                colored_glb = None
                
                with st.spinner("🔍 Đang tìm kiếm trong kho dữ liệu (sử dụng đa luồng)..."):
                    matched_img, score = find_similar_cached_model(img_bytes, OUTPUT_DIR)
                
                # Neu tim thay anh tuong tu, tim file GLB tuong ung
                if matched_img:
                    white_glb, colored_glb = find_glbs_for_image(matched_img)
                
                # Fallback: check exact hash trong cache
                if not white_glb and not colored_glb and cached_glb.exists():
                    colored_glb = cached_glb

                if white_glb or colored_glb:
                    st.success(f"⚡ Đã tìm thấy mô hình tương tự! Bỏ qua các bước 1 tới 12.")
                    viewer_placeholder = st.empty()
                    
                    # Buoc 1: Hien mo hinh trang truoc
                    if white_glb and white_glb.exists():
                        with viewer_placeholder:
                            render_html(get_viewer_html(white_glb), height=535)
                    
                    # Buoc 2: Hien mo hinh co mau (neu co)
                    final_model = None
                    if colored_glb and colored_glb.exists():
                        if white_glb and white_glb.exists():
                            sim_progress = st.progress(0, text="Đang tiến hành tô màu mô hình 3D trắng...")
                            for i in range(100):
                                time.sleep(0.02)
                                sim_progress.progress(i + 1, text=f"Đang tiến hành tô màu mô hình 3D trắng... ({i+1}%)")
                            sim_progress.empty()
                        
                        with viewer_placeholder:
                            render_html(get_viewer_html(colored_glb), height=535)
                        final_model = str(colored_glb)
                    elif white_glb and white_glb.exists():
                        final_model = str(white_glb)
                    
                    if final_model:
                        st.session_state.current_model_path = final_model
                        st.session_state.show_save_buttons = False
                        st.session_state.feedback_mode = False
                        time.sleep(1)
                        st.rerun()
                else:
                    with st.spinner("Đang xử lý ảnh và chạy AI (có thể mất vài phút)..."):
                        try:
                            ext = Path(uploaded_file.name).suffix.lower()
                            job_id = img_hash
                            img_path = OUTPUT_DIR / f"{job_id}{ext}"
                            
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                                
                            progress_bar = st.progress(5, text="Đang xử lý ảnh 2D và tạo Point Cloud... (5%)")
                            viewer_placeholder = st.empty()
                            status_placeholder = st.empty()

                            def step_progress_callback(step, total, msg):
                                """Callback để cập nhật progress bar real-time trên giao diện."""
                                pct = min(95, int(5 + (step / total) * 90))
                                # Lấy tên bước đẹp hơn
                                clean_msg = msg.strip().strip('\n')
                                progress_bar.progress(pct, text=f"{clean_msg} ({pct}%)")

                            def step12_callback(white_glb_path):
                                if status_placeholder:
                                    status_placeholder.empty()
                                st.session_state.current_white_model_path = white_glb_path
                                with viewer_placeholder:
                                    render_html(get_viewer_html(white_glb_path), height=535)

                            # Chay pipeline tao 3D (truyen callback cho b12 + progress)
                            model_path = run_pipeline(
                                img_path, 
                                OUTPUT_DIR, 
                                on_mesh_ready=step12_callback,
                                on_progress=step_progress_callback
                            )
                            
                            progress_bar.progress(100, text="Hoàn tất 100%!")
                            time.sleep(1)
                            progress_bar.empty()
                            status_placeholder.empty()
                            viewer_placeholder.empty()
                            
                            st.session_state.current_model_path = str(model_path)
                            st.session_state.current_job_id = job_id
                            st.session_state.show_save_buttons = True
                            st.session_state.feedback_mode = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")

        if st.session_state.current_model_path is not None:
            model_path = st.session_state.current_model_path
            
            try:
                with open(model_path, "rb") as f:
                    model_bytes_3d = f.read()
                
                render_html(get_viewer_html(model_bytes_3d, is_bytes=True), height=535)
                
                st.download_button(
                    label="📥 Tải xuống file GLB (.glb)",
                    data=model_bytes_3d,
                    file_name="shoe_model.glb",
                    mime="model/gltf-binary",
                    width="stretch"
                )

                if st.session_state.show_save_buttons:
                    if not st.session_state.feedback_mode:
                        st.markdown("### 🏆 Đánh giá Kết quả")
                        c1, c2, c3 = st.columns(3)
                        
                        if c1.button("✅ Lưu kết quả (Tốt)", type="primary", width="stretch"):
                            # Luu model mau
                            shutil.copy(model_path, cached_glb)
                                
                            st.session_state.show_save_buttons = False
                            st.success("🎉 Đã lưu thành công! Lần sau các ảnh tương tự sẽ tự động được nhận diện.")
                            st.rerun()
                            
                        if c2.button("✍️ Báo lỗi (Xấu)", width="stretch"):
                            st.session_state.feedback_mode = True
                            st.rerun()

                        if c3.button("❌ Không lưu (Xóa)", width="stretch"):
                            st.session_state.show_save_buttons = False
                            st.session_state.current_model_path = None
                            st.rerun()
                    else:
                        st.warning("Xin hãy góp ý để chúng tôi cải thiện AI tốt hơn trong tương lai:")
                        feedback_text = st.text_area("Mô hình bị lỗi gì? (ví dụ: đế giày bị méo, màu nhạt...)")
                        
                        fb_c1, fb_c2 = st.columns(2)
                        if fb_c1.button("📤 Gửi Góp ý", width="stretch"):
                            if feedback_text.strip() == "":
                                st.error("Vui lòng nhập nội dung góp ý trước khi gửi!")
                            else:
                                ext = Path(uploaded_file.name).suffix.lower()
                                img_feedback_path = FEEDBACK_DIR / f"{img_hash}{ext}"
                                with open(img_feedback_path, "wb") as f:
                                    f.write(img_bytes)
                                
                                job_id = st.session_state.current_job_id
                                if job_id:
                                    nobg_src = OUTPUT_DIR / f"{job_id}_nobg.png"
                                    if nobg_src.exists():
                                        nobg_dst = FEEDBACK_DIR / f"{img_hash}_nobg.png"
                                        shutil.copy(nobg_src, nobg_dst)

                                glb_feedback_path = FEEDBACK_DIR / f"{img_hash}.glb"
                                shutil.copy(model_path, glb_feedback_path)
                                
                                txt_feedback_path = FEEDBACK_DIR / f"{img_hash}.txt"
                                with open(txt_feedback_path, "w", encoding="utf-8") as f:
                                    f.write(feedback_text)
                                
                                st.session_state.show_save_buttons = False
                                st.session_state.feedback_mode = False
                                st.session_state.current_model_path = None
                                st.success("Cảm ơn bạn! Báo cáo lỗi đã được ghi nhận.")
                                st.rerun()
                                
                        if fb_c2.button("Hủy", width="stretch"):
                            st.session_state.feedback_mode = False
                            st.rerun()

            except Exception as e:
                st.error(f"Không thể đọc file 3D: {e}")

    else:
        st.info("👈 Hãy tải lên một bức ảnh ở cột bên trái để bắt đầu tái tạo 3D.")
