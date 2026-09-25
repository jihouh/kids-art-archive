import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import json
from datetime import date

# Page Setup
st.set_page_config(page_title="My Art Book 🎨", layout="wide", initial_sidebar_state="collapsed")

# File Storage Setup
IMAGE_DIR = "processed_art"
DB_FILE = "art_metadata.json"
os.makedirs(IMAGE_DIR, exist_ok=True)

if not os.path.exists(DB_FILE):
    with open(DB_FILE, "w") as f:
        json.dump([], f)

def load_metadata():
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_metadata(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

# --- IMAGE PROCESSING ENGINE ---
def auto_white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.add(a, int(128 - np.mean(a)))
    b = cv2.add(b, int(128 - np.mean(b)))
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def enhance_art(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
    
    pil_img = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Color(pil_img).enhance(1.25)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(1.1)
    return pil_img

def process_upload(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    balanced = auto_white_balance(img)
    final_pil = enhance_art(balanced)
    
    filename = f"art_{uploaded_file.name}"
    save_path = os.path.join(IMAGE_DIR, filename)
    final_pil.save(save_path)
    return filename, save_path

# --- CLEAN CSS OVERRIDES ---
st.markdown("""
    <style>
    /* Overall Background */
    .stApp {
        background-color: #F7F3E9;
    }
    
    /* Global Font & Header */
    h1, h2, h3, p, div {
        font-family: 'Comic Sans MS', 'Chalkboard SE', 'Fredoka', cursive, sans-serif !important;
    }
    
    /* Clean Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        border-radius: 10px 10px 0 0 !important;
    }
    
    /* Big Kid Buttons */
    div.stButton > button {
        width: 100% !important;
        height: 60px !important;
        font-size: 1.4rem !important;
        font-weight: bold !important;
        border-radius: 15px !important;
        border: 2px solid #5A3E2B !important;
        box-shadow: 0px 4px 0px #5A3E2B !important;
        cursor: pointer !important;
    }
    div.stButton > button:active {
        transform: translateY(3px) !important;
        box-shadow: 0px 1px 0px #5A3E2B !important;
    }
    </style>
""", unsafe_allow_html=True)

# Application Title
st.title("📖 My Art Book 🎨")

tab1, tab2, tab3 = st.tabs(["📖 Read Book", "🖼️ View Gallery", "➕ Add New Page"])

# --- TAB 1: STORYBOOK MODE ---
with tab1:
    metadata = load_metadata()
    if not metadata:
        st.info("The art book is currently empty! Click 'Add New Page' to upload artwork.")
    else:
        if "slide_idx" not in st.session_state:
            st.session_state.slide_idx = 0

        if st.session_state.slide_idx >= len(metadata):
            st.session_state.slide_idx = 0

        current_art = metadata[st.session_state.slide_idx]

        # Book Border Wrapper
        with st.container(border=True):
            col_left, col_spine, col_right = st.columns([1.1, 0.05, 0.9])

            # Left Page: Artwork
            with col_left:
                st.image(current_art["file_path"], use_container_width=True)

            # Center Spine Visual
            with col_spine:
                st.markdown("<div style='border-right: 3px dashed #D1C2A5; height: 100%; min-height: 400px; margin: 0 auto;'></div>", unsafe_allow_html=True)

            # Right Page: Metadata & Kid Controls
            with col_right:
                st.markdown(f"## **{current_art['title']}**")
                st.markdown(f"📅 **Date:** {current_art['date']}")
                st.markdown("---")
                st.markdown(f"**Story:**")
                st.info(current_art['description'])
                
                st.write("")
                st.write("")

                # Large Touch Buttons
                btn_left, btn_right = st.columns(2)
                with btn_left:
                    if st.button("⬅️ BACK", key="prev_page"):
                        st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(metadata)
                        st.rerun()

                with btn_right:
                    if st.button("NEXT ➡️", key="next_page"):
                        st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(metadata)
                        st.rerun()

                st.markdown(f"<h4 style='text-align: center; color: #7F5A3C; margin-top: 15px;'>Page {st.session_state.slide_idx + 1} of {len(metadata)}</h4>", unsafe_allow_html=True)

# --- TAB 2: GALLERY VIEW ---
with tab2:
    metadata = load_metadata()
    if metadata:
        cols = st.columns(3)
        for idx, item in enumerate(metadata):
            with cols[idx % 3]:
                with st.container(border=True):
                    st.image(item["file_path"], use_container_width=True)
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"{item['date']}")

# --- TAB 3: UPLOAD & NEW PAGE ---
with tab3:
    st.header("Add a New Page to the Book")
    uploaded_file = st.file_uploader("Snap or select a photo of the artwork", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        title = st.text_input("Artwork Title", value="My Masterpiece")
        art_date = st.date_input("Date Created", value=date.today())
        desc = st.text_area("Description / Story", value="Made with paint and love!")
        
        if st.button("✨ Clean Up & Add Page"):
            filename, save_path = process_upload(uploaded_file)
            
            metadata = load_metadata()
            metadata.append({
                "filename": filename,
                "file_path": save_path,
                "title": title,
                "date": str(art_date),
                "description": desc
            })
            save_metadata(metadata)
            st.success("Page added to the book!")
            st.balloons()
