"""Tests for the PDF processor module."""
import io
from unittest import mock

import pytest
from PyPDF2 import PdfReader

from pdf_processor import extract_text_from_pdf


class TestExtractTextFromPdf:
    """Tests for extract_text_from_pdf function."""

    @pytest.fixture
    def mock_pdf_reader(self):
        """Create a mock PDF reader."""
        with mock.patch("PyPDF2.PdfReader") as mock_reader:
            # Mock pages
            mock_page1 = mock.MagicMock()
            mock_page1.extract_text.return_value = "Page 1 content"

            mock_page2 = mock.MagicMock()
            mock_page2.extract_text.return_value = "Page 2 content"

            # Set up reader to return mock pages
            mock_reader.return_value.pages = [mock_page1, mock_page2]
            yield mock_reader

    def test_extract_from_file_path(self, mock_pdf_reader):
        """Test extracting text from a PDF file path."""
        with mock.patch("builtins.open", mock.mock_open()) as mock_file:
            result = extract_text_from_pdf("test.pdf")

            mock_file.assert_called_once_with("test.pdf", "rb")
            mock_pdf_reader.assert_called_once()
            assert result == "Page 1 content\nPage 2 content\n"

    def test_extract_from_bytes_io(self, mock_pdf_reader):
        """Test extracting text from a BytesIO object."""
        bytes_io = io.BytesIO(b"fake pdf content")
        result = extract_text_from_pdf(bytes_io)

        mock_pdf_reader.assert_called_once_with(bytes_io)
        assert result == "Page 1 content\nPage 2 content\n"
