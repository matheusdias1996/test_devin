"""Streamlit app for PDF entity extraction and summarization."""
import json
import os
from typing import Dict, Any, Optional

import streamlit as st
# Import required modules
from entity_extractor import EntityExtractor
from css_loader import load_css

def main():
    """Run the Streamlit app."""
    # Configure page settings
    st.set_page_config(
        page_title="PDF Entity Extractor & Summarizer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    try:
        load_css("static/style.css")
    except Exception as e:
        st.warning(f"Could not load custom CSS: {str(e)}")
    
    # Create header with gradient background
    st.markdown(
        """
        <div class="header-container">
            <h1>PDF Entity Extractor & Summarizer</h1>
            <p>Upload a PDF to extract entities or generate a summary</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Get API key from user input
    api_key = st.text_input("Enter your Google API Key:", type="password")
    if not api_key:
        st.warning("Please enter your Google API Key to use the app")
        st.stop()

    # Create a container for the file uploader
    with st.container():
        st.markdown(
            """
            <div class="upload-container">
                <h3>📁 Upload Your Document</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["🔍 Entity Extraction", "📝 Summarization"])
    
    with tab1:
        st.markdown("<h3>Extract Entities from Your Document</h3>", unsafe_allow_html=True)
        
        # Create two columns for input and instructions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Entity types input
            entity_types_input = st.text_area(
                "Enter entity types to extract (one per line)",
                "Name\nDate\nAddress\nPhone Number",
            )
        
        with col2:
            st.markdown("""
                <div class="info-box">
                    <h4>💡 Tips</h4>
                    <p>Enter each entity type on a new line. Examples: Name, Date, Address, Phone Number, Email, etc.</p>
                </div>
                """, unsafe_allow_html=True)
            
        # Extract button with enhanced styling
        extract_button = st.button("🔍 Extract Entities", key="extract_entities", use_container_width=True)
        
    with tab2:
        st.markdown("<h3>Generate a Summary of Your Document</h3>", unsafe_allow_html=True)
        
        # Create two columns for input and instructions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Summarization options
            max_length = st.number_input(
                "Maximum summary length (words, leave at 0 for no limit)",
                min_value=0,
                value=200,
                step=50,
            )
            max_length = None if max_length == 0 else max_length
        
        with col2:
            st.markdown("""
                <div class="info-box">
                    <h4>💡 Tips</h4>
                    <p>Set a maximum length for your summary or leave at 0 for no limit.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Summarize button with enhanced styling
        summarize_button = st.button("📝 Generate Summary", key="generate_summary", use_container_width=True)
    
    # Process extraction request
    if extract_button and uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Show upload success
            with st.expander("File Upload"):
                st.success("PDF uploaded successfully")

            # Parse entity types
            entity_types = [
                line.strip() for line in entity_types_input.split("\n") if line.strip()
            ]

            # Extract entities
            extractor = EntityExtractor(api_key)
            try:
                entities_json = extractor.extract_entities(uploaded_file, entity_types)

                # Parse JSON if it's a string
                if isinstance(entities_json, str):
                    entities = json.loads(entities_json)
                else:
                    entities = entities_json

                # Display results
                st.markdown(
                    """
                    <div class="result-header">
                        <h2>📊 Extracted Entities</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display JSON in a styled container
                with st.container():
                    st.json(entities)

                # Allow downloading results
                st.download_button(
                    label="Download Results as JSON",
                    data=json.dumps(entities, indent=2),
                    file_name="extracted_entities.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"Error extracting entities: {str(e)}")
    
    # Process summarization request
    if summarize_button and uploaded_file is not None:
        with st.spinner("Generating summary..."):
            # Show upload success
            with st.expander("File Upload"):
                st.success("PDF uploaded successfully")
            
            # Generate summary
            extractor = EntityExtractor(api_key)
            try:
                summary = extractor.summarize_text(uploaded_file, max_length)
                
                # Display results
                st.markdown(
                    """
                    <div class="result-header">
                        <h2>📝 Summary</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display summary in a styled container with red text
                st.markdown(f"<div class='summary-text' style='color: red;'>{summary}</div>", unsafe_allow_html=True)
                
                # Allow downloading summary
                st.download_button(
                    label="Download Summary as Text",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")


if __name__ == "__main__":
    main()
