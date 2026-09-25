import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import json
from datetime import date

# Set page layout to wide
st.set_page_config(page_title="My Art Book 🎨", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM DIGITAL BOOK CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Fredoka', cursive, sans-serif;
        background-color: #f4ece1 !important;
    }
    
    .main {
        background-color: #f4ece1 !important;
    }

    /* Book Frame Container */
    .book-container {
        background-color: #fffdf9;
        border: 12px solid #8b5a2b;
        border-radius: 24px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.2), inset 0 0 15px rgba(0,0,0,0.08);
        padding: 30px;
        margin-top: 10px;
        position: relative;
    }
    
    /* Center Spine Effect */
    .book-spine {
        border-right: 4px dashed #d1c2a5;
        padding-right: 25px;
    }
    
    .book-page-right {
        padding-left: 25px;
    }

    /* Artwork Paper Card */
    .art-frame {
        background: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.12);
        border: 1px solid #e2d7c5;
    }

    /* Typography */
    .book-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #5c3a21;
        text-align: center;
        margin-bottom: 5px;
    }

    .art-title {
        font-size: 2rem;
        color: #2c3e50;
        font-weight: 700;
        margin-top: 15px;
    }

    .art-meta {
        font-size: 1.1rem;
        color: #7f8c8d;
    }

    .art-desc {
        font-size: 1.2rem;
        color: #34495e;
        background-color: #fcf8f2;
        padding: 12px;
        border-radius: 10px;
        border-left: 5px solid #e67e22;
        margin-top: 10px;
    }

    /* Kid-Friendly Chunky Buttons */
    .stButton>button {
        font-family: 'Fredoka', cursive;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        border-radius: 18px !important;
        padding: 12px 20px !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 6px 0px rgba(0,0,0,0.2) !important;
        transition: all 0.1s ease !important;
    }
    
    .stButton>button:active {
        transform: translateY(4px) !important;
        box-shadow: 0 2px 0px rgba(0,0,0,0.2) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        border-radius: 12px 12px 0 0 !important;
        background-color: #e6d7c3 !important;
        padding: 10px 20px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #8b5a2b !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

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

# --- HEADER ---
st.markdown("<div class='book-title'>📖 My Art Book 🎨</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📖 Read Art Book", "🖼️ View Gallery", "➕ Add New Page"])

# --- TAB 1: TWO-PAGE DIGITAL BOOK MODE ---
with tab1:
    metadata = load_metadata()
    if not metadata:
        st.info("The art book is currently empty! Click 'Add New Page' to upload the first artwork.")
    else:
        if "slide_idx" not in st.session_state:
            st.session_state.slide_idx = 0

        if st.session_state.slide_idx >= len(metadata):
            st.session_state.slide_idx = 0

        current_art = metadata[st.session_state.slide_idx]

        # Render Open Book Container
        st.markdown("<div class='book-container'>", unsafe_allow_html=True)
        left_col, right_col = st.columns([1.1, 0.9])

        # Left Page: Artwork Display
        with left_col:
            st.markdown("<div class='book-spine'>", unsafe_allow_html=True)
            st.image(current_art["file_path"], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Right Page: Title, Description & Big Controls
        with right_col:
            st.markdown("<div class='book-page-right'>", unsafe_allow_html=True)
            st.markdown(f"<div class='art-title'>{current_art['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='art-meta'><b>Date:</b> {current_art['date']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='art-desc'><b>Story:</b><br>{current_art['description']}</div>", unsafe_allow_html=True)
            
            st.write("")
            st.write("")
            
            # Kid Controls (Previous, Next)
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("🔴 BACK", key="back_btn"):
                    st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(metadata)
                    st.rerun()
            with btn_col2:
                if st.button("🟢 NEXT", key="next_btn"):
                    st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(metadata)
                    st.rerun()

            st.markdown(f"<h4 style='text-align: center; color: #8b5a2b; margin-top: 15px;'>Page {st.session_state.slide_idx + 1} of {len(metadata)}</h4>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: GALLERY VIEW ---
with tab2:
    metadata = load_metadata()
    if metadata:
        cols = st.columns(3)
        for idx, item in enumerate(metadata):
            with cols[idx % 3]:
                st.markdown("<div class='art-frame'>", unsafe_allow_html=True)
                st.image(item["file_path"], use_container_width=True)
                st.markdown(f"<b>{item['title']}</b><br><small>{item['date']}</small>", unsafe_allow_html=True)
                st.markdown("</div><br>", unsafe_allow_html=True)

# --- TAB 3: UPLOAD & NEW PAGE ---
with tab3:
    st.header("Add a New Page to the Book")
    uploaded_file = st.file_uploader("Snap or select a photo of the artwork", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        title = st.text_input("Artwork Title", value="My Masterpiece")
        art_date = st.date_input("Date Created", value=date.today())
        desc = st.text_area("Description / What is happening in this picture?", value="I drew this with colored pencils!")
        
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
