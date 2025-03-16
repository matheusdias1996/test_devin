"""Streamlit app for PDF entity extraction and summarization."""
import json
import os
from typing import Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

from entity_extractor import EntityExtractor
from pdf_processor import extract_text_from_pdf

# Load environment variables
load_dotenv()


def main():
    """Run the Streamlit app."""
    st.title("PDF Entity Extractor & Summarizer")
    st.write("Upload a PDF to extract entities or generate a summary")

    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set GOOGLE_API_KEY in your .env file")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Entity Extraction", "Summarization"])
    
    with tab1:
        # Entity types input
        entity_types_input = st.text_area(
            "Enter entity types to extract (one per line)",
            "Name\nDate\nAddress\nPhone Number",
        )
        
        extract_button = st.button("Extract Entities", key="extract_entities")
        
    with tab2:
        # Summarization options
        max_length = st.number_input(
            "Maximum summary length (words, leave at 0 for no limit)",
            min_value=0,
            value=200,
            step=50,
        )
        max_length = None if max_length == 0 else max_length
        
        summarize_button = st.button("Generate Summary", key="generate_summary")
    
    # Process extraction request
    if extract_button and uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)

            # Show extracted text
            with st.expander("Extracted Text"):
                st.text(pdf_text)

            # Parse entity types
            entity_types = [
                line.strip() for line in entity_types_input.split("\n") if line.strip()
            ]

            # Extract entities
            extractor = EntityExtractor(api_key)
            try:
                entities_json = extractor.extract_entities(pdf_text, entity_types)

                # Parse JSON if it's a string
                if isinstance(entities_json, str):
                    entities = json.loads(entities_json)
                else:
                    entities = entities_json

                # Display results
                st.subheader("Extracted Entities")
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
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(uploaded_file)
            
            # Show extracted text
            with st.expander("Extracted Text"):
                st.text(pdf_text)
            
            # Generate summary
            extractor = EntityExtractor(api_key)
            try:
                summary = extractor.summarize_text(pdf_text, max_length)
                
                # Display results
                st.subheader("Summary")
                st.markdown(f"<div style='color: red;'>{summary}</div>", unsafe_allow_html=True)
                
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
