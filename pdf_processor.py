"""Module for processing PDF files."""
import io
from typing import Union

import PyPDF2


def extract_text_from_pdf(pdf_file: Union[str, io.BytesIO]) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_file: Path to PDF file or BytesIO object containing PDF data

    Returns:
        Extracted text from the PDF
    """
    text = ""
    
    if isinstance(pdf_file, str):
        # Open from file path
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    else:
        # Process BytesIO object
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
            
    return text 