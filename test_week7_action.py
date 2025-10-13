"""
Test script for Week 7 Action Recognition Module
"""

import streamlit as st
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.week07.action_recognition_module import ActionRecognitionModule


def main():
    st.set_page_config(
        page_title="Week 7: Action Recognition Test",
        page_icon="🎬",
        layout="wide"
    )

    # Initialize the module
    module = ActionRecognitionModule()

    # Render the module
    module.render()


if __name__ == "__main__":
    main()