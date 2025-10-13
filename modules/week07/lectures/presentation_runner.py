"""
Week 7 Lecture Presentation Runner
Interactive slide viewer for lecture materials
"""

import streamlit as st
import os
from pathlib import Path

def load_slides():
    """Load lecture slides from markdown file"""
    slides_path = Path(__file__).parent / "lecture_slides.md"

    with open(slides_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by slide separator (---)
    slides = content.split('\n---\n')
    return slides

def main():
    st.set_page_config(
        page_title="Week 7: Action Recognition - Lecture",
        page_icon="🎬",
        layout="wide"
    )

    # Custom CSS for presentation style
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background: white;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem;
    }
    h1 {
        color: #2d3748;
        font-size: 2.5rem !important;
        text-align: center;
        border-bottom: 3px solid #667eea;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    h2 {
        color: #4a5568;
        font-size: 2rem !important;
        margin-top: 1.5rem;
    }
    h3 {
        color: #718096;
        font-size: 1.5rem !important;
    }
    .stButton > button {
        background: #667eea;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: #5a67d8;
        transform: scale(1.05);
    }
    code {
        background: #f7fafc;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        color: #e53e3e;
    }
    pre {
        background: #2d3748;
        color: #fff;
        padding: 1rem;
        border-radius: 5px;
        overflow-x: auto;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    th {
        background: #667eea;
        color: white;
        padding: 0.75rem;
        text-align: left;
    }
    td {
        padding: 0.75rem;
        border-bottom: 1px solid #e2e8f0;
    }
    blockquote {
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        color: #4a5568;
        font-style: italic;
        margin: 1rem 0;
    }
    .slide-number {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'slide_index' not in st.session_state:
        st.session_state.slide_index = 0

    # Load slides
    slides = load_slides()
    total_slides = len(slides)

    # Sidebar for navigation
    with st.sidebar:
        st.title("📑 Navigation")

        # Slide selector
        st.session_state.slide_index = st.selectbox(
            "Go to slide:",
            range(total_slides),
            format_func=lambda x: f"Slide {x+1}/{total_slides}",
            index=st.session_state.slide_index
        )

        st.markdown("---")

        # Quick links
        st.subheader("🔗 Quick Links")

        if st.button("🏠 Title Slide"):
            st.session_state.slide_index = 0

        if st.button("📚 Part 1: Overview"):
            st.session_state.slide_index = 3

        if st.button("🏗️ Part 2: Core Tech"):
            st.session_state.slide_index = 7

        if st.button("🔧 Part 3: MediaPipe"):
            st.session_state.slide_index = 12

        if st.button("☁️ Part 4: Google API"):
            st.session_state.slide_index = 20

        if st.button("🧪 Part 5: Labs"):
            st.session_state.slide_index = 27

        if st.button("📊 Part 6: Comparison"):
            st.session_state.slide_index = 34

        if st.button("📝 Part 7: Projects"):
            st.session_state.slide_index = 37

        st.markdown("---")

        # Lab launchers
        st.subheader("🚀 Quick Launch")

        if st.button("▶️ Run MediaPipe Demo"):
            st.info("Run in terminal:\n```\ncd modules/week07/labs\npython lab06_mediapipe_google_demo.py\n```")

        if st.button("▶️ Run Streamlit App"):
            st.info("Run in terminal:\n```\nstreamlit run test_week7_action.py\n```")

        st.markdown("---")

        # Resources
        st.subheader("📚 Resources")
        st.markdown("""
        - [MediaPipe Docs](https://google.github.io/mediapipe/)
        - [Google Video API](https://cloud.google.com/video-intelligence/docs)
        - [Week 7 GitHub](https://github.com/your-repo)
        """)

    # Main content area
    col1, col2, col3 = st.columns([1, 8, 1])

    with col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.slide_index == 0)):
            st.session_state.slide_index = max(0, st.session_state.slide_index - 1)
            st.rerun()

    with col3:
        if st.button("Next ➡️", disabled=(st.session_state.slide_index == total_slides - 1)):
            st.session_state.slide_index = min(total_slides - 1, st.session_state.slide_index + 1)
            st.rerun()

    # Display current slide
    with col2:
        current_slide = slides[st.session_state.slide_index]

        # Process slide content
        if st.session_state.slide_index == 0:
            # Special formatting for title slide
            st.markdown(f"<div style='text-align: center; padding: 3rem 0;'>{current_slide}</div>",
                       unsafe_allow_html=True)
        else:
            # Regular slide
            st.markdown(current_slide)

        # Add slide number
        st.markdown(
            f"<div class='slide-number'>Slide {st.session_state.slide_index + 1} / {total_slides}</div>",
            unsafe_allow_html=True
        )

    # Keyboard shortcuts info
    with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
        st.markdown("""
        - **←/→**: Navigate slides
        - **Home**: First slide
        - **End**: Last slide
        - **F**: Fullscreen (F11 in browser)
        - **Space**: Next slide
        """)

    # Progress bar
    progress = (st.session_state.slide_index + 1) / total_slides
    st.progress(progress)

    # Footer with additional info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Week 7**: Action Recognition")

    with col2:
        st.markdown("**Instructor**: AI Vision Course")

    with col3:
        st.markdown("**Duration**: 90 minutes")

if __name__ == "__main__":
    main()