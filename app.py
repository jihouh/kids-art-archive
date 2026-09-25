import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import json
from datetime import date

# Set page layout to wide and child-friendly title
st.set_page_config(page_title="My Art Gallery 🎨", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for big child-friendly touch controls
st.markdown("""
    <style>
    .big-title { font-size: 3rem !important; font-weight: bold; color: #FF4B4B; text-align: center; }
    .stButton>button { font-size: 1.5rem !important; border-radius: 20px !important; padding: 10px 25px !important; width: 100%; }
    .art-card { background-color: #F0F2F6; padding: 15px; border-radius: 15px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Folder settings
IMAGE_DIR = "processed_art"
DB_FILE = "art_metadata.json"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Initialize JSON storage
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
    # Read image buffer into numpy array safely
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process
    balanced = auto_white_balance(img)
    final_pil = enhance_art(balanced)
    
    filename = f"art_{uploaded_file.name}"
    save_path = os.path.join(IMAGE_DIR, filename)
    final_pil.save(save_path)
    return filename, save_path

# --- NAVIGATION TABS ---
st.markdown("<div class='big-title'>🎨 My Digital Art Gallery 🎨</div>", unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3 = st.tabs(["🖼️ Slideshow & Play", "📚 Gallery View", "📤 Add New Art"])

# --- TAB 1: CHILD-FRIENDLY SLIDESHOW ---
with tab1:
    metadata = load_metadata()
    if not metadata:
        st.info("No artwork added yet! Go to 'Add New Art' tab to upload the first picture.")
    else:
        if "slide_idx" not in st.session_state:
            st.session_state.slide_idx = 0

        # Prevent out-of-bounds index
        if st.session_state.slide_idx >= len(metadata):
            st.session_state.slide_idx = 0

        current_art = metadata[st.session_state.slide_idx]
        
        # Display image with updated parameter
        st.image(current_art["file_path"], use_container_width=True)
        st.markdown(f"### **{current_art['title']}** ({current_art['date']})")
        st.write(f"*{current_art['description']}*")
        
        # Big Navigation Controls
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ PREVIOUS"):
                st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(metadata)
                st.rerun()
        with col2:
            st.write(f"**Item {st.session_state.slide_idx + 1} of {len(metadata)}**")
        with col3:
            if st.button("NEXT ➡️"):
                st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(metadata)
                st.rerun()

# --- TAB 2: GALLERY VIEW ---
with tab2:
    metadata = load_metadata()
    if metadata:
        cols = st.columns(3)
        for idx, item in enumerate(metadata):
            with cols[idx % 3]:
                st.image(item["file_path"], use_container_width=True)
                st.caption(f"**{item['title']}** - {item['date']}")

# --- TAB 3: UPLOAD & CLEANUP ---
with tab3:
    st.header("Upload New Artwork")
    uploaded_file = st.file_uploader("Choose a photo of the artwork", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        title = st.text_input("Artwork Title", value="My Masterpiece")
        art_date = st.date_input("Date Created", value=date.today())
        desc = st.text_area("Description", value="Made with paint and love!")
        
        if st.button("✨ Clean Up & Save Artwork"):
            filename, save_path = process_upload(uploaded_file)
            
            # Save metadata
            metadata = load_metadata()
            metadata.append({
                "filename": filename,
                "file_path": save_path,
                "title": title,
                "date": str(art_date),
                "description": desc
            })
            save_metadata(metadata)
            st.success("Artwork digitized and added to your collection!")
            st.balloons()
