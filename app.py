import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
import zipfile
import tempfile
import pandas as pd
import plotly.express as px

# Import your custom modules
from ai_core import load_ai_model
from math_core import recover_image
from data_utils import save_to_database, calculate_clarity_gain, generate_pdf_report, DB_FILE

st.set_page_config(layout="wide", page_title="VisionClear AI Dashboard")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    div[data-testid="metric-container"] { background: linear-gradient(135deg, #f6f8fd 0%, #f1f5f9 100%); border-radius: 10px; padding: 15px; border-left: 6px solid #FF4B4B; box-shadow: 2px 4px 10px rgba(0,0,0,0.1); }
    div[data-testid="stMetric"]:nth-child(2) div[data-testid="metric-container"] { border-left: 6px solid #00D2D3; }
    div[data-testid="stMetric"]:nth-child(3) div[data-testid="metric-container"] { border-left: 6px solid #FDCB6E; }
</style>
""", unsafe_allow_html=True)

if "uploader_key" not in st.session_state: st.session_state.uploader_key = "initial_key"
if "session_count" not in st.session_state: st.session_state.session_count = 0
if "session_data" not in st.session_state: st.session_state.session_data = []

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103322.png", width=60)
    st.title("Control Panel")
    st.markdown("---")
    method = st.radio("⚙️ Processing Engine:", ["Advanced Baseline (Guided DCP)", "AI Deep Learning (DehazeNet)"])
    omega = 0.95
    if method == "Advanced Baseline (Guided DCP)": omega = st.slider("🎚️ De-hazing Strength", 0.0, 1.0, 0.95, 0.05)
    st.markdown("---")
    if st.button("🔄 Reset Workspace", use_container_width=True):
        st.session_state.uploader_key = str(time.time())
        st.session_state.session_count = 0
        st.session_state.session_data = []
        st.rerun()
    st.markdown("---")
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as file: st.download_button(label="📥 Export Report", data=file, file_name="Report.csv", mime="text/csv")

st.title("☁️ AI-Based Image De-smoking and De-hazing System for Enhanced Computer Vision")
st.markdown("**Center for Emerging Computing Technologies - Major Project**")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Images Processed (Current Session)", st.session_state.session_count)
col_b.metric("Active Engine", "DehazeNet CNN" if "AI" in method else "Guided DCP Math")
col_c.metric("System Status", "Online & Ready")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📸 Single Image & Webcam", "🗂️ Batch Processing", "🎥 Video Processing", "📊 Live Analytics"])

with tab1:
    input_source = st.radio("Select Input Source:", ["Upload File", "Use Webcam"], horizontal=True)
    uploaded_file = st.file_uploader("📂 Upload...", type=["jpg", "jpeg", "png"], key=st.session_state.uploader_key) if input_source == "Upload File" else st.camera_input("📷 Take a picture")
    
    if uploaded_file is not None and st.button("🚀 Process Image", use_container_width=True):
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        start_time = time.time()
        with st.spinner(f"Executing {method}..."):
            ai_model = load_ai_model() if "AI" in method else None
            recovered_img = recover_image(img_array, method, omega, ai_model)
            exec_time = time.time() - start_time
            clarity_gain = calculate_clarity_gain(img_array, recovered_img)
            save_to_database(uploaded_file.name if hasattr(uploaded_file, 'name') else "webcam_capture", method, exec_time, clarity_gain)
            st.session_state.session_count += 1
        
        st.success(f"✅ Complete in {exec_time:.2f}s | Clarity +{clarity_gain}%")
        col1, col2 = st.columns(2)
        with col1: st.image(image, use_container_width=True, caption="Original Input")
        with col2: st.image(recovered_img, use_container_width=True, caption="De-hazed Output")
                
        result_pil = Image.fromarray(recovered_img)
        os.makedirs("Saved_Results", exist_ok=True)
        filename = uploaded_file.name if hasattr(uploaded_file, 'name') else "webcam_capture.jpg"
        result_pil.save(f"Saved_Results/clean_{filename}", format="JPEG")
        
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG")
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1: st.download_button("💾 Save Clean Image", buf.getvalue(), "clean_image.jpg", "image/jpeg", use_container_width=True)
        with col_btn2: 
            pdf_bytes = generate_pdf_report(image, result_pil, filename, method, exec_time, clarity_gain)
            st.download_button("📄 Download Case Report", pdf_bytes, f"Report_{filename}.pdf", "application/pdf", use_container_width=True)

with tab2:
    batch_files = st.file_uploader("Upload batch images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if batch_files and st.button("Start Batch Processing"):
        progress_bar = st.progress(0)
        zip_buffer = io.BytesIO()
        ai_model = load_ai_model() if "AI" in method else None
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, file in enumerate(batch_files):
                img_arr = np.array(Image.open(file).convert("RGB"))
                t0 = time.time()
                rec_img = recover_image(img_arr, method, omega, ai_model)
                save_to_database(file.name, method, time.time() - t0, calculate_clarity_gain(img_arr, rec_img))
                rec_pil = Image.fromarray(rec_img)
                buf = io.BytesIO()
                rec_pil.save(buf, format='JPEG')
                zf.writestr(f"clean_{file.name}", buf.getvalue())
                progress_bar.progress((i + 1) / len(batch_files))
                st.session_state.session_count += 1
        st.success("✅ Batch processing complete!")
        st.download_button("📥 Download All (ZIP)", zip_buffer.getvalue(), "Dehazed_Batch.zip", "application/zip")

with tab3:
    video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
    if video_file and st.button("Process Video"):
        ai_model = load_ai_model() if "AI" in method else None
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(out_file.name, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        video_progress, frame_count, total = st.progress(0), 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            out.write(cv2.cvtColor(recover_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), method, omega, ai_model), cv2.COLOR_RGB2BGR))
            frame_count += 1
            video_progress.progress(min(frame_count / total, 1.0))
            
        cap.release()
        out.release()
        with open(out_file.name, 'rb') as vf:
            v_bytes = vf.read()
            st.video(v_bytes)
            st.download_button("💾 Download Video", v_bytes, "dehazed.mp4", "video/mp4")

with tab4:
    if "session_data" in st.session_state and len(st.session_state.session_data) > 0:
        df = pd.DataFrame(st.session_state.session_data)
        k1, k2, k3 = st.columns(3)
        k1.metric("🖼️ Images Processed", len(df))
        k2.metric("📈 Avg Clarity Gain", f"+{df['Clarity Numeric'].mean():.1f}%")
        k3.metric("⏱️ Avg Process Time", f"{df['Time (s)'].mean():.2f}s")
        st.write("") 
        col_g1, col_g2 = st.columns(2)
        with col_g1: st.plotly_chart(px.pie(df, names='Method', title='Engine Usage', hole=0.5).update_layout(height=280, margin=dict(t=35,b=0,l=0,r=0)), use_container_width=True)
        with col_g2: st.plotly_chart(px.bar(df, x='File', y='Clarity Numeric', title='Clarity Gain (%)', color='Method').update_layout(height=280, margin=dict(t=35,b=0,l=0,r=0)), use_container_width=True)
        st.plotly_chart(px.area(df, x=df.index, y='Time (s)', title='Speed Trend', markers=True).update_layout(height=220, margin=dict(t=35,b=0,l=0,r=0)), use_container_width=True)
    else: st.info("Process an image to generate analytics!")

st.markdown("---")
with st.expander("🕒 Watch History (Persistent Database)"):
    if os.path.exists(DB_FILE):
        history_df = pd.read_csv(DB_FILE).sort_values(by="Date", ascending=False)
        col_db1, col_db2 = st.columns([4, 1])
        with col_db1: st.info("👆 Click on any row below to view its specific output record.")
        with col_db2:
            if st.button("🗑️ Clear History", use_container_width=True): os.remove(DB_FILE); st.rerun()
        
        event = st.dataframe(history_df, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
        if len(event.selection.rows) > 0:
            data = history_df.iloc[event.selection.rows[0]]
            st.markdown("---")
            st.success(f"✅ Complete in {data['Time (s)']}s | Clarity {data['Clarity Gain']}")
            rec_img_col, rec_col1, rec_col2 = st.columns([1, 2, 2])
            with rec_img_col:
                img_name = str(data['File']) if str(data['File']).lower().endswith(('.jpg', '.jpeg', '.png')) else str(data['File']) + ".jpg"
                img_path = os.path.join("Saved_Results", f"clean_{img_name}")
                if os.path.exists(img_path):
                    st.image(img_path, width=150)
                    with st.popover("🔍 Full Size"): st.image(img_path, use_container_width=True)
            with rec_col1: st.metric("File Name", data['File']); st.metric("Date", data['Date'])
            with rec_col2: st.metric("Engine", data['Method'])