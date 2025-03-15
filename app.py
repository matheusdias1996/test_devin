"""Streamlit app for PDF entity extraction."""
import json
import os
from typing import Dict, Any

import streamlit as st
from dotenv import load_dotenv

from entity_extractor import EntityExtractor
from pdf_processor import extract_text_from_pdf

# Load environment variables
load_dotenv()


def main():
    """Run the Streamlit app."""
    st.title("PDF Entity Extractor")
    st.write("Upload a PDF and specify entities to extract")

    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set GOOGLE_API_KEY in your .env file")
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Entity types input
    entity_types_input = st.text_area(
        "Enter entity types to extract (one per line)",
        "Name\nDate\nAddress\nPhone Number",
    )

    if st.button("Extract Entities") and uploaded_file is not None:
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


if __name__ == "__main__":
    main()
