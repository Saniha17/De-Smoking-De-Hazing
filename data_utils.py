import pandas as pd
from datetime import datetime
import os
import cv2
import numpy as np
import tempfile
from fpdf import FPDF
import streamlit as st

DB_FILE = "processing_history.csv"

def save_to_database(filename, method, exec_time, clarity):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{"Date": now, "File": filename, "Method": method, "Time (s)": round(exec_time, 2), "Clarity Gain": f"+{clarity}%"}])
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        df = pd.concat([df, new_data], ignore_index=True)
    else:
        df = new_data
    df.to_csv(DB_FILE, index=False)
    
    if "session_data" not in st.session_state:
        st.session_state.session_data = []
    st.session_state.session_data.append({
        "File": filename,
        "Method": method,
        "Time (s)": round(exec_time, 2),
        "Clarity Numeric": float(clarity)
    })

def calculate_clarity_gain(hazy_img, clean_img):
    hazy_std = np.std(cv2.cvtColor(hazy_img, cv2.COLOR_RGB2GRAY))
    clean_std = np.std(cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY))
    if hazy_std == 0: return 0
    gain = ((clean_std - hazy_std) / hazy_std) * 100
    return round(max(0, gain), 1)

def generate_pdf_report(orig_pil, clean_pil, filename, method, exec_time, clarity):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "VisionClear AI - Project Case Report", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"File Name: {filename}", ln=True)
    pdf.cell(0, 8, f"Processing Engine: {method}", ln=True)
    pdf.cell(0, 8, f"Execution Time: {exec_time:.2f} seconds", ln=True)
    pdf.cell(0, 8, f"Clarity Improvement: +{clarity}%", ln=True)
    pdf.cell(0, 8, f"Date Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f_orig, \
         tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f_clean:
         
        orig_pil.save(f_orig.name, format="JPEG")
        clean_pil.save(f_clean.name, format="JPEG")
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Visual Analysis:", ln=True)
        
        y_pos = pdf.get_y()
        pdf.image(f_orig.name, x=10, y=y_pos, w=90)
        pdf.image(f_clean.name, x=110, y=y_pos, w=90)
        
        pdf.set_y(y_pos + 70)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(90, 10, "Original Hazy Image", align='C')
        pdf.cell(90, 10, "De-hazed Output", align='C', ln=True)
        
    pdf_out = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(pdf_out.name)
    
    with open(pdf_out.name, "rb") as f:
        pdf_bytes = f.read()
        
    return pdf_bytes