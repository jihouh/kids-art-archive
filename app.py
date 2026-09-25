import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import json
from datetime import date
from google import genai

# Page Setup
st.set_page_config(page_title="My Art Book 🎨", layout="centered", initial_sidebar_state="collapsed")

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

# --- GEMINI AI IMAGE ANALYSIS ENGINE ---
def analyze_artwork_with_gemini(pil_image):
    """Uses Gemini to generate a creative title and short story for children's artwork."""
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Fallback if no API Key is set
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        
    if not api_key:
        st.warning("⚠️ GEMINI_API_KEY not found. Using default title and description.")
        return "My Masterpiece", "Made with paint and love!"

    try:
        client = genai.Client(api_key=api_key)
        
        prompt = (
            "Analyze this child's artwork or craft piece. Provide a JSON response with exactly two keys:\n"
            "1. 'title': A short, fun, creative title suitable for a child's art book (max 5 words).\n"
            "2. 'description': A warm, encouraging 1-2 sentence story describing what is shown in the artwork.\n"
            "Respond ONLY with valid JSON."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[pil_image, prompt],
            config={
                'response_mime_type': 'application/json'
            }
        )
        
        result = json.loads(response.text)
        return result.get("title", "My Masterpiece"), result.get("description", "Made with paint and love!")
        
    except Exception as e:
        st.error(f"AI Analysis Error: {e}")
        return "My Masterpiece", "Made with paint and love!"

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
    return filename, save_path, final_pil

# --- CROSS-PLATFORM & iOS SAFARI CSS FIXES ---
st.markdown("""
    <style>
    :root {
        color-scheme: light !important;
    }

    .stApp {
        background-color: #F7F3E9 !important;
        color: #2C2C2C !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #3D2314 !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        color: #3D2314 !important;
        margin-bottom: 12px;
    }

    .page-counter {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 800;
        color: #8B4513 !important;
        line-height: 48px;
    }

    [data-testid="column"] {
        width: 33.33% !important;
        flex: 1 1 33.33% !important;
        min-width: auto !important;
    }

    div.stButton > button {
        width: 100% !important;
        height: 48px !important;
        background-color: #4A2E1B !important;
        color: #FFFFFF !important;
        font-size: 1rem !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0px 3px 0px #2C1A0E !important;
        -webkit-appearance: none !important;
    }
    
    div.stButton > button p {
        color: #FFFFFF !important;
        font-size: 1rem !important;
        font-weight: bold !important;
    }

    div.stButton > button:active {
        transform: translateY(2px) !important;
        box-shadow: 0px 1px 0px #2C1A0E !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E6D7C3 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 8px 12px !important;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #3D2314 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 16px !important;
        padding: 12px !important;
        border: 2px solid #E2D7C5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<div class='main-title'>📖 My Art Book 🎨</div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📖 Read Book", "🖼️ Gallery", "➕ Add Page"])

# --- TAB 1: SINGLE-PAGE STORYBOOK ---
with tab1:
    metadata = load_metadata()
    if not metadata:
        st.info("The art book is currently empty! Click 'Add Page' to upload artwork.")
    else:
        if "slide_idx" not in st.session_state:
            st.session_state.slide_idx = 0

        if st.session_state.slide_idx >= len(metadata):
            st.session_state.slide_idx = 0

        current_art = metadata[st.session_state.slide_idx]

        with st.container(border=True):
            nav_left, nav_mid, nav_right = st.columns([1, 1, 1])
            
            with nav_left:
                if st.button("⬅️ BACK", key="prev_page_single"):
                    st.session_state.slide_idx = (st.session_state.slide_idx - 1) % len(metadata)
                    st.rerun()

            with nav_mid:
                st.markdown(f"<div class='page-counter'>{st.session_state.slide_idx + 1} / {len(metadata)}</div>", unsafe_allow_html=True)

            with nav_right:
                if st.button("NEXT ➡️", key="next_page_single"):
                    st.session_state.slide_idx = (st.session_state.slide_idx + 1) % len(metadata)
                    st.rerun()

            st.write("")

            st.image(current_art["file_path"], use_container_width=True)

            st.write("")

            st.markdown(f"### **{current_art['title']}**")
            st.markdown(f"📅 **Date:** {current_art['date']}")
            st.markdown("---")
            st.markdown("**Story / Description:**")
            st.info(current_art['description'])

# --- TAB 2: GALLERY VIEW ---
with tab2:
    metadata = load_metadata()
    if metadata:
        cols = st.columns(2)
        for idx, item in enumerate(metadata):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.image(item["file_path"], use_container_width=True)
                    st.markdown(f"**{item['title']}**")
                    st.caption(f"{item['date']}")

# --- TAB 3: UPLOAD & NEW PAGE WITH AI ANALYSIS ---
with tab3:
    st.header("Add a New Page")
    uploaded_file = st.file_uploader("Snap or select a photo of the artwork", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Check if this image has been processed by AI yet
        if "analyzed_filename" not in st.session_state or st.session_state.analyzed_filename != uploaded_file.name:
            with st.spinner("🤖 AI is reading the artwork and crafting a story..."):
                filename, save_path, pil_img = process_upload(uploaded_file)
                ai_title, ai_desc = analyze_artwork_with_gemini(pil_img)
                
                # Store in session state for form pre-filling
                st.session_state.analyzed_filename = uploaded_file.name
                st.session_state.temp_filename = filename
                st.session_state.temp_save_path = save_path
                st.session_state.generated_title = ai_title
                st.session_state.generated_desc = ai_desc

        # Pre-filled Editable Form
        title = st.text_input("Artwork Title", value=st.session_state.get("generated_title", "My Masterpiece"))
        art_date = st.date_input("Date Created", value=date.today())
        desc = st.text_area("Description / Story", value=st.session_state.get("generated_desc", "Made with paint and love!"))
        
        if st.button("✨ Clean Up & Add Page"):
            metadata = load_metadata()
            metadata.append({
                "filename": st.session_state.temp_filename,
                "file_path": st.session_state.temp_save_path,
                "title": title,
                "date": str(art_date),
                "description": desc
            })
            save_metadata(metadata)
            
            # Clear state after successful save
            del st.session_state["analyzed_filename"]
            
            st.success("Page added to the book!")
            st.balloons()
