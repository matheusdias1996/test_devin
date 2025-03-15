"""Extended tests for the PDF processor module."""
import io
from unittest import mock

import pytest
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

from pdf_processor import extract_text_from_pdf


class TestExtractTextFromPdfExtended:
    """Extended tests for extract_text_from_pdf function."""

    def test_extract_from_empty_pdf(self):
        """Test extracting text from an empty PDF."""
        with mock.patch("builtins.open", mock.mock_open()), \
             mock.patch("PyPDF2.PdfReader") as mock_reader:
            # Mock empty PDF (no pages)
            mock_reader.return_value.pages = []
            
            result = extract_text_from_pdf("empty.pdf")
            
            assert result == ""

    def test_extract_with_file_not_found(self):
        """Test handling of FileNotFoundError."""
        with mock.patch("builtins.open") as mock_open:
            mock_open.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(FileNotFoundError):
                extract_text_from_pdf("nonexistent.pdf")

    def test_extract_with_invalid_pdf(self):
        """Test handling of invalid PDF file."""
        with mock.patch("builtins.open", mock.mock_open()), mock.patch("PyPDF2.PdfReader") as mock_reader:
            mock_reader.side_effect = PdfReadError("Invalid PDF file")
            
            with pytest.raises(PdfReadError):
                extract_text_from_pdf("invalid.pdf")

    def test_extract_with_empty_pages(self):
        """Test extracting text from PDF with empty pages."""
        with mock.patch("builtins.open", mock.mock_open()), \
             mock.patch("PyPDF2.PdfReader") as mock_reader:
            # Mock pages with empty text
            mock_page1 = mock.MagicMock()
            mock_page1.extract_text.return_value = ""
            
            mock_page2 = mock.MagicMock()
            mock_page2.extract_text.return_value = ""
            
            # Set up reader to return mock pages
            mock_reader.return_value.pages = [mock_page1, mock_page2]
            
            result = extract_text_from_pdf("empty_pages.pdf")
            
            assert result == "\n\n"

    def test_extract_with_permission_error(self):
        """Test handling of permission error when opening file."""
        with mock.patch("builtins.open") as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                extract_text_from_pdf("protected.pdf")

    def test_extract_with_bytes_io_empty(self):
        """Test extracting text from an empty BytesIO object."""
        with mock.patch("PyPDF2.PdfReader") as mock_reader:
            # Mock empty PDF (no pages)
            mock_reader.return_value.pages = []
            
            bytes_io = io.BytesIO(b"")
            result = extract_text_from_pdf(bytes_io)
            
            assert result == ""

    def test_extract_with_bytes_io_invalid(self):
        """Test handling of invalid BytesIO content."""
        with mock.patch("PyPDF2.PdfReader") as mock_reader:
            mock_reader.side_effect = PdfReadError("Invalid PDF content")
            
            bytes_io = io.BytesIO(b"not a pdf")
            
            with pytest.raises(PdfReadError):
                extract_text_from_pdf(bytes_io)
