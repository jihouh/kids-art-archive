import os
import json
import time
from datetime import date
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from google import genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="My Art Book 🎨", layout="centered", initial_sidebar_state="collapsed")

# Initialize Active Navigation State
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "📖 Read Book"

# --- CUSTOM CSS: CLEAN UI ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Comic Neue', cursive, sans-serif !important;
        background-color: #d1b894 !important;
    }
    
    .main {
        background-color: #d1b894 !important;
        padding-top: 10px !important;
    }

    /* Header Title */
    .app-header {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #2b1a0e;
        margin-bottom: 15px;
    }

    /* Storybook Typography */
    .art-title-text {
        font-size: 2.2rem;
        font-weight: 700;
        color: #000000;
        margin-top: 10px;
        margin-bottom: 4px;
        line-height: 1.1;
    }

    .art-date-text {
        font-size: 1.1rem;
        color: #333333;
        margin-bottom: 8px;
    }

    .art-desc-text {
        font-size: 1.25rem;
        color: #111111;
        line-height: 1.35;
        background-color: #fdfaf3;
        padding: 12px 15px;
        border-radius: 10px;
        border-left: 4px solid #6b4423;
        margin-bottom: 10px;
    }

    .page-counter-top {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 700;
        color: #4a3319;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* CHUNKY 3D BUTTON STYLING */
    div.stButton > button {
        font-family: 'Comic Neue', cursive !important;
        width: 100% !important;
        min-height: 65px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        border-radius: 18px !important;
        border: none !important;
        color: #ffffff !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: transform 0.1s ease !important;
    }

    div.stButton > button p {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }

    div.stButton > button:active {
        transform: translateY(4px) !important;
        box-shadow: none !important;
    }

    /* Color Specific 3D Action Buttons */
    /* Back Button (Red) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) div.stButton > button {
        background-color: #d94338 !important;
        box-shadow: 0 6px 0 #9e2a22 !important;
    }

    /* Next Button (Green) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) div.stButton > button {
        background-color: #63b33a !important;
        box-shadow: 0 6px 0 #437d26 !important;
    }

    /* See All / Gallery Button (Blue) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(3) div.stButton > button {
        background-color: #2b84cb !important;
        box-shadow: 0 6px 0 #1b588a !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- FILE STORAGE INITIALIZATION ---
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

# --- OPENCV IMAGE PROCESSING ---
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
    return final_pil, filename, save_path

# --- GEMINI AI ANALYSIS ---
def analyze_artwork_with_gemini(pil_image, max_retries=3):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        
    if not api_key:
        return "Rainbow Dino", "Me and a dino under the rainbow!"

    client = genai.Client(api_key=api_key)
    prompt = (
        "Analyze this child's artwork. Provide a JSON response with two keys:\n"
        "1. 'title': A short, playful title (max 4 words).\n"
        "2. 'description': A simple, cheerful sentence describing the drawing.\n"
        "Respond ONLY in valid JSON."
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[pil_image, prompt],
                config={'response_mime_type': 'application/json'}
            )
            result = json.loads(response.text)
            return result.get("title", "Rainbow Dino"), result.get("description", "Me and a dino under the rainbow!")
        except Exception:
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + 1)
                continue
            return "Rainbow Dino", "Me and a dino under the rainbow!"

# --- HEADER ---
st.markdown("<div class='app-header'>My Art Book 🎨</div>", unsafe_allow_html=True)

# Custom Top Navigation Pills synced with Session State
selected_tab = st.pills(
    options=["📖 Read Book", "🖼️ Gallery", "➕ Add Page"],
    label="",
    selection_mode="single",
    default=st.session_state.current_tab,
    key="nav_pills"
)

# Update session state if pill selection changes
if selected_tab and selected_tab != st.session_state.current_tab:
    st.session_state.current_tab = selected_tab
    st.rerun()

# --- VIEW 1: SINGLE-PAGE STORYBOOK VIEW ---
if st.session_state.current_tab == "📖 Read Book":
    metadata = load_metadata()
    
    if not metadata:
        st.info("Your art book is empty right now. Switch to '➕ Add Page' to upload artwork!")
    else:
        if "slide_idx" not in st.session_state:
            st.session_state.slide_idx = 0

        if st.session_state.slide_idx >= len(metadata):
            st.session_state.slide_idx = 0

        current_art = metadata[st.session_state.slide_idx]

        # 1. TOP NAVIGATION BUTTONS (BACK, NEXT, SEE ALL)
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("⬅️\nBACK", key="top_back_btn"):
                st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(metadata)
                st.rerun()

        with btn_col2:
            if st.button("➡️\nNEXT", key="top_next_btn"):
                st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(metadata)
                st.rerun()

        with btn_col3:
            if st.button("🖼️\nSEE ALL", key="top_see_all_btn"):
                st.session_state.current_tab = "🖼️ Gallery"
                st.rerun()

        # 2. PAGE COUNTER
        st.markdown(f"<div class='page-counter-top'>Page {st.session_state.slide_idx + 1} of {len(metadata)}</div>", unsafe_allow_html=True)

        # 3. DIRECT ARTWORK DISPLAY
        st.image(current_art["file_path"], use_container_width=True)

        # 4. TITLE, DATE, AND DESCRIPTION BELOW ARTWORK
        st.markdown(f"<div class='art-title-text'>{current_art['title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='art-date-text'><b>Date:</b> {current_art['date']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='art-desc-text'><b>Description:</b> {current_art['description']}</div>", unsafe_allow_html=True)

# --- VIEW 2: GALLERY VIEW ---
elif st.session_state.current_tab == "🖼️ Gallery":
    metadata = load_metadata()
    if not metadata:
        st.info("No artwork in gallery yet. Add some pages first!")
    else:
        st.markdown("### All Artwork Pages 🎨")
        cols = st.columns(2)
        for idx, item in enumerate(metadata):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.image(item["file_path"], use_container_width=True)
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"Date: {item['date']}")
                    
                    # Clicking opens this item in book view
                    if st.button(f"📖 Read Page {idx+1}", key=f"open_art_{idx}"):
                        st.session_state.slide_idx = idx
                        st.session_state.current_tab = "📖 Read Book"
                        st.rerun()

# --- VIEW 3: UPLOAD & NEW PAGE ---
elif st.session_state.current_tab == "➕ Add Page":
    st.header("Add Artwork to the Book")
    uploaded_file = st.file_uploader("Choose a photo of the artwork", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        art_date = st.date_input("Date Created", value=date.today())
        
        if st.button("✨ Clean Up & Add Page"):
            with st.spinner("Processing image and crafting story..."):
                final_pil, filename, save_path = process_upload(uploaded_file)
                ai_title, ai_desc = analyze_artwork_with_gemini(final_pil)
                
                metadata = load_metadata()
                metadata.append({
                    "filename": filename,
                    "file_path": save_path,
                    "title": ai_title,
                    "date": str(art_date),
                    "description": ai_desc
                })
                save_metadata(metadata)
                st.session_state.slide_idx = len(metadata) - 1
                st.session_state.current_tab = "📖 Read Book"
                st.success("Added to your art book!")
                st.balloons()
                st.rerun()
