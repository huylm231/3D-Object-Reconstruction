import streamlit as st
import sys
import os
import uuid
import base64
import hashlib
import shutil
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

from src.pipeline import run_pipeline

# ──────────────────── Giao dien Streamlit ────────────────────
st.set_page_config(page_title="3D Shoe Reconstruction", page_icon="👟", layout="wide")

st.title("👟 3D Shoe Reconstruction")
st.markdown("Tải lên một bức ảnh 2D của giày và AI sẽ tái tạo thành mô hình 3D tương tác!")

# Chia cot cho giao dien
col1, col2 = st.columns(2)

with col1:
    st.header("Ảnh Đầu Vào 2D")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh (JPG, PNG)", type=['png', 'jpg', 'jpeg', 'webp'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh bạn đã tải lên", use_container_width=True)

with col2:
    st.header("Mô Hình 3D")
    
    if uploaded_file is not None:
        img_bytes = uploaded_file.getvalue()
        # Tao ma bam cho anh de nhan dien
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        cached_glb = CACHE_DIR / f"{img_hash}.glb"

        # Khoi tao trang thai cho Streamlit
        if "last_hash" not in st.session_state or st.session_state.last_hash != img_hash:
            st.session_state.last_hash = img_hash
            st.session_state.current_model_path = None
            st.session_state.show_save_buttons = False
            st.session_state.feedback_mode = False
            st.session_state.current_job_id = None

        # 1. Hien thi nut bat dau neu chua co ket qua
        if st.session_state.current_model_path is None:
            if st.button("🚀 Bắt đầu tạo mô hình 3D", type="primary"):
                # Kiem tra xem da co trong cache chua
                if cached_glb.exists():
                    st.success("⚡ Ảnh này đã có trong dữ liệu! Tải mô hình ngay lập tức...")
                    st.session_state.current_model_path = cached_glb
                    st.session_state.show_save_buttons = False
                    st.session_state.feedback_mode = False
                    st.rerun()
                else:
                    with st.spinner("Đang xử lý ảnh và chạy AI (có thể mất vài phút)..."):
                        try:
                            ext = Path(uploaded_file.name).suffix.lower()
                            job_id = str(uuid.uuid4())
                            img_path = UPLOAD_DIR / f"{job_id}{ext}"
                            
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                                
                            # Chay pipeline tao 3D
                            model_path = run_pipeline(img_path, OUTPUT_DIR)
                            
                            st.session_state.current_model_path = model_path
                            st.session_state.current_job_id = job_id
                            st.session_state.show_save_buttons = True
                            st.session_state.feedback_mode = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")

        # 2. Hien thi 3D Viewer neu da co ket qua
        if st.session_state.current_model_path is not None:
            model_path = st.session_state.current_model_path
            
            try:
                with open(model_path, "rb") as f:
                    model_bytes_3d = f.read()
                
                b64_model = base64.b64encode(model_bytes_3d).decode('utf-8')
                model_data_uri = f"data:model/gltf-binary;base64,{b64_model}"
                
                viewer_html = f"""
                <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
                <div style="background-color: #1E1E1E; border-radius: 10px; overflow: hidden; padding: 10px; margin-bottom: 15px;">
                    <model-viewer 
                        src="{model_data_uri}" 
                        alt="3D Shoe Model" 
                        auto-rotate 
                        camera-controls 
                        shadow-intensity="1" 
                        style="width: 100%; height: 500px; outline: none; background-color: #1E1E1E;">
                    </model-viewer>
                </div>
                """
                st.components.v1.html(viewer_html, height=535)
                
                st.download_button(
                    label="📥 Tải xuống file GLB",
                    data=model_bytes_3d,
                    file_name="shoe_model.glb",
                    mime="model/gltf-binary"
                )

                # 3. Hien thi 3 nut bam cho mo hinh moi tao
                if st.session_state.show_save_buttons:
                    if not st.session_state.feedback_mode:
                        st.info("Mô hình AI tạo ra có làm bạn hài lòng không?")
                        c1, c2, c3 = st.columns(3)
                        
                        if c1.button("✅ Lưu kết quả (Tốt)", type="primary"):
                            shutil.copy(model_path, cached_glb)
                            st.session_state.show_save_buttons = False
                            st.success("Đã lưu thành công vào CSDL! Lần sau tải ảnh này lên sẽ ra kết quả ngay lập tức.")
                            st.rerun()
                            
                        if c2.button("✍️ Báo lỗi (Xấu)"):
                            st.session_state.feedback_mode = True
                            st.rerun()

                        if c3.button("❌ Không lưu (Xoá)"):
                            st.session_state.show_save_buttons = False
                            st.session_state.current_model_path = None
                            st.rerun()
                    else:
                        # 4. Hien thi Form Ghi Chu Loi
                        st.warning("Xin hãy góp ý để chúng tôi cải thiện AI tốt hơn trong tương lai:")
                        feedback_text = st.text_area("Mô hình bị lỗi gì? (ví dụ: đế giày bị méo, màu nhạt...)")
                        
                        fb_c1, fb_c2 = st.columns(2)
                        if fb_c1.button("📤 Gửi Góp ý"):
                            if feedback_text.strip() == "":
                                st.error("Vui lòng nhập nội dung góp ý trước khi gửi!")
                            else:
                                ext = Path(uploaded_file.name).suffix.lower()
                                
                                # 1. Luu 1 anh goc duy nhat (chong de bang ma bam)
                                img_feedback_path = FEEDBACK_DIR / f"{img_hash}{ext}"
                                with open(img_feedback_path, "wb") as f:
                                    f.write(img_bytes)
                                
                                # 2. Luu 1 anh tach nen duy nhat
                                job_id = st.session_state.current_job_id
                                if job_id:
                                    nobg_src = OUTPUT_DIR / f"{job_id}_nobg.png"
                                    if nobg_src.exists():
                                        nobg_dst = FEEDBACK_DIR / f"{img_hash}_nobg.png"
                                        shutil.copy(nobg_src, nobg_dst)

                                # 3. Luu 1 file 3D GLB duy nhat
                                glb_feedback_path = FEEDBACK_DIR / f"{img_hash}.glb"
                                shutil.copy(model_path, glb_feedback_path)
                                
                                # 4. Luu noi dung loi
                                txt_feedback_path = FEEDBACK_DIR / f"{img_hash}.txt"
                                with open(txt_feedback_path, "w", encoding="utf-8") as f:
                                    f.write(feedback_text)
                                
                                st.session_state.show_save_buttons = False
                                st.session_state.feedback_mode = False
                                st.session_state.current_model_path = None
                                st.success("Cảm ơn bạn! Báo cáo lỗi (gồm Ảnh gốc, Ảnh tách nền, Mô hình 3D và Góp ý) đã được ghi nhận thành công.")
                                st.rerun()
                                
                        if fb_c2.button("Hủy"):
                            st.session_state.feedback_mode = False
                            st.rerun()

            except Exception as e:
                st.error(f"Khong the doc file 3D: {e}")

    else:
        st.info("👈 Hãy tải lên một bức ảnh ở cột bên trái để bắt đầu.")
