import streamlit as st

def load_css(css_file):
    """Load CSS styles from a file."""
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
