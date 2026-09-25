import os
import json
import time
from datetime import date
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from google import genai

# --- PAGE CONFIGURATION (Centered Single Page) ---
st.set_page_config(page_title="My Art Book 🎨", layout="centered", initial_sidebar_state="collapsed")

# --- CUSTOM SINGLE-COLUMN CSS STYLING ---
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

    /* Single Page Card Frame */
    .single-page-card {
        background-color: #fffdf9;
        border: 8px solid #8b5a2b;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* Typography */
    .book-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #5c3a21;
        text-align: center;
        margin-bottom: 10px;
    }

    .art-title {
        font-size: 1.8rem;
        color: #2c3e50;
        font-weight: 700;
        margin-top: 15px;
        text-align: center;
    }

    .art-meta {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 10px;
    }

    .art-desc {
        font-size: 1.15rem;
        color: #34495e;
        background-color: #fcf8f2;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #e67e22;
        margin-top: 10px;
    }

    .page-counter {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        color: #8b5a2b;
        padding-top: 8px;
    }

    /* Chunky Navigation Buttons */
    .stButton>button {
        font-family: 'Fredoka', cursive;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border-radius: 16px !important;
        padding: 10px 15px !important;
        width: 100% !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 0px rgba(0,0,0,0.2) !important;
        transition: all 0.1s ease !important;
    }
    
    .stButton>button:active {
        transform: translateY(3px) !important;
        box-shadow: 0 1px 0px rgba(0,0,0,0.2) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 10px 10px 0 0 !important;
        background-color: #e6d7c3 !important;
        padding: 8px 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #8b5a2b !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- FILE STORAGE SETUP ---
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

# --- GEMINI AI ANALYSIS FUNCTION ---
def analyze_artwork_with_gemini(pil_image, max_retries=3):
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        
    if not api_key:
        st.warning("⚠️ GEMINI_API_KEY not found. Using default title and description.")
        return "My Masterpiece", "Made with paint and love!"

    client = genai.Client(api_key=api_key)
    prompt = (
        "Analyze this child's artwork or craft piece. Provide a JSON response with exactly two keys:\n"
        "1. 'title': A short, fun, creative title suitable for a child's art book (max 5 words).\n"
        "2. 'description': A warm, encouraging 1-2 sentence story describing what is shown in the artwork.\n"
        "Respond ONLY with valid JSON."
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=[pil_image, prompt],
                config={
                    'response_mime_type': 'application/json'
                }
            )
            
            result = json.loads(response.text)
            return result.get("title", "My Masterpiece"), result.get("description", "Made with paint and love!")
            
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1
                    time.sleep(wait_time)
                    continue
            
            st.warning("⚠️ Gemini server is temporarily busy. Applied standard template so you can keep going!")
            return "My Masterpiece", "Made with paint and love!"

# --- UI HEADER ---
st.markdown("<div class='book-title'>📖 My Art Book 🎨</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📖 Read Art Book", "🖼️ View Gallery", "➕ Add New Page"])

# --- TAB 1: SINGLE-COLUMN PAGE MODE ---
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

        st.markdown("<div class='single-page-card'>", unsafe_allow_html=True)
        
        # 1. NAVIGATION BAR ABOVE IMAGE
        nav_col1, nav_col2, nav_col3 = st.columns([1, 1.2, 1])
        with nav_col1:
            if st.button("⬅️ BACK", key="single_back"):
                st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(metadata)
                st.rerun()
        with nav_col2:
            st.markdown(f"<div class='page-counter'>Page {st.session_state.slide_idx + 1} of {len(metadata)}</div>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("NEXT ➡️", key="single_next"):
                st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(metadata)
                st.rerun()

        st.write("")

        # 2. CENTERED IMAGE
        st.image(current_art["file_path"], use_container_width=True)

        st.write("")

        # 3. TITLE, DATE, AND STORY BELOW IMAGE
        st.markdown(f"<div class='art-title'>{current_art['title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='art-meta'>Date: {current_art['date']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='art-desc'><b>Story:</b><br>{current_art['description']}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: GALLERY VIEW ---
with tab2:
    metadata = load_metadata()
    if metadata:
        cols = st.columns(2)
        for idx, item in enumerate(metadata):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.image(item["file_path"], use_container_width=True)
                    st.markdown(f"<b>{item['title']}</b><br><small>{item['date']}</small>", unsafe_allow_html=True)

# --- TAB 3: UPLOAD & NEW PAGE ---
with tab3:
    st.header("Add a New Page to the Book")
    uploaded_file = st.file_uploader("Snap or select a photo of the artwork", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        art_date = st.date_input("Date Created", value=date.today())
        
        if st.button("✨ Auto-Clean & Analyze with AI"):
            with st.spinner("Cleaning image and asking Gemini AI for story ideas..."):
                final_pil, filename, save_path = process_upload(uploaded_file)
                ai_title, ai_description = analyze_artwork_with_gemini(final_pil)
                
                st.session_state.temp_art = {
                    "filename": filename,
                    "save_path": save_path,
                    "title": ai_title,
                    "description": ai_description,
                    "date": str(art_date)
                }

        if "temp_art" in st.session_state:
            st.subheader("Review Page Details")
            title = st.text_input("Artwork Title", value=st.session_state.temp_art["title"])
            desc = st.text_area("Story / Description", value=st.session_state.temp_art["description"])
            
            if st.button("📖 Save to Art Book"):
                metadata = load_metadata()
                metadata.append({
                    "filename": st.session_state.temp_art["filename"],
                    "file_path": st.session_state.temp_art["save_path"],
                    "title": title,
                    "date": st.session_state.temp_art["date"],
                    "description": desc
                })
                save_metadata(metadata)
                del st.session_state["temp_art"]
                st.success("Page added to the book!")
                st.balloons()
