"""Tests for the Streamlit app."""
import json
import os
from unittest import mock

import pytest
import streamlit as st

# Import the main function from app.py
from app import main


class TestApp:
    """Tests for the Streamlit app."""

    @pytest.fixture
    def mock_streamlit(self):
        """Mock Streamlit components."""
        with mock.patch("streamlit.title"), \
             mock.patch("streamlit.write"), \
             mock.patch("streamlit.error"), \
             mock.patch("streamlit.stop"), \
             mock.patch("streamlit.file_uploader"), \
             mock.patch("streamlit.text_area"), \
             mock.patch("streamlit.button"), \
             mock.patch("streamlit.spinner"), \
             mock.patch("streamlit.expander"), \
             mock.patch("streamlit.text"), \
             mock.patch("streamlit.subheader"), \
             mock.patch("streamlit.json"), \
             mock.patch("streamlit.download_button"):
            yield

    @pytest.fixture
    def mock_env_with_api_key(self):
        """Mock environment with API key."""
        with mock.patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "fake-api-key"
            yield mock_getenv

    @pytest.fixture
    def mock_env_without_api_key(self):
        """Mock environment without API key."""
        with mock.patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = None
            yield mock_getenv

    def test_main_without_api_key(self, mock_streamlit, mock_env_without_api_key):
        """Test main function without API key."""
        with mock.patch("streamlit.stop") as mock_stop, \
             mock.patch("streamlit.file_uploader") as mock_uploader:
            # Ensure file_uploader returns None to avoid PDF processing
            mock_uploader.return_value = None
            main()
            mock_stop.assert_called_once()

    def test_main_with_api_key_no_file(self, mock_streamlit, mock_env_with_api_key):
        """Test main function with API key but no file uploaded."""
        with mock.patch("streamlit.file_uploader") as mock_uploader, \
             mock.patch("streamlit.button") as mock_button:
            
            # Mock no file uploaded
            mock_uploader.return_value = None
            
            # Mock button click
            mock_button.return_value = True
            
            main()
            
            # No processing should happen without a file

    def test_main_with_file_and_extraction(self, mock_streamlit, mock_env_with_api_key):
        """Test main function with file upload and entity extraction."""
        # Set up environment variable for Google API
        with mock.patch.dict(os.environ, {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "false"}):
            with mock.patch("app.extract_text_from_pdf") as mock_extract, \
                 mock.patch("app.EntityExtractor") as mock_extractor_class, \
                 mock.patch("streamlit.file_uploader") as mock_uploader, \
                 mock.patch("streamlit.button") as mock_button, \
                 mock.patch("streamlit.spinner") as mock_spinner, \
                 mock.patch("streamlit.expander") as mock_expander, \
                 mock.patch("streamlit.json") as mock_json, \
                 mock.patch("streamlit.download_button") as mock_download:
                
                # Mock file upload
                mock_file = mock.MagicMock()
                mock_uploader.return_value = mock_file
                
                # Mock button click
                mock_button.return_value = True
                
                # Mock PDF extraction
                mock_extract.return_value = "Extracted PDF text"
                
                # Mock spinner and expander contexts
                mock_spinner.return_value.__enter__.return_value = None
                mock_spinner.return_value.__exit__.return_value = None
                mock_expander.return_value.__enter__.return_value = None
                mock_expander.return_value.__exit__.return_value = None
                
                # Mock entity extractor
                mock_extractor = mock.MagicMock()
                mock_extractor_class.return_value = mock_extractor
                mock_extractor.extract_entities.return_value = json.dumps({"Name": "John Doe"})
                
                # Call main function
                with mock.patch("streamlit.text") as mock_text, \
                     mock.patch("streamlit.subheader") as mock_subheader:
                    main()
                
                # Verify PDF extraction was called
                mock_extract.assert_called_once_with(mock_file)
                
                # Verify entity extraction was called
                mock_extractor.extract_entities.assert_called_once()

    def test_main_with_extraction_error(self, mock_streamlit, mock_env_with_api_key):
        """Test main function with extraction error."""
        # Set up environment variable for Google API
        with mock.patch.dict(os.environ, {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "false"}):
            with mock.patch("app.extract_text_from_pdf") as mock_extract, \
                 mock.patch("app.EntityExtractor") as mock_extractor_class, \
                 mock.patch("streamlit.file_uploader") as mock_uploader, \
                 mock.patch("streamlit.button") as mock_button, \
                 mock.patch("streamlit.spinner") as mock_spinner, \
                 mock.patch("streamlit.expander") as mock_expander, \
                 mock.patch("streamlit.error") as mock_error:
                
                # Mock file upload
                mock_file = mock.MagicMock()
                mock_uploader.return_value = mock_file
                
                # Mock button click
                mock_button.return_value = True
                
                # Mock PDF extraction
                mock_extract.return_value = "Extracted PDF text"
                
                # Mock spinner and expander contexts
                mock_spinner.return_value.__enter__.return_value = None
                mock_spinner.return_value.__exit__.return_value = None
                mock_expander.return_value.__enter__.return_value = None
                mock_expander.return_value.__exit__.return_value = None
                
                # Mock entity extractor with error
                mock_extractor = mock.MagicMock()
                mock_extractor_class.return_value = mock_extractor
                mock_extractor.extract_entities.side_effect = Exception("Extraction error")
                
                # Call main function
                with mock.patch("streamlit.text") as mock_text, \
                     mock.patch("streamlit.subheader") as mock_subheader:
                    main()
                
                # Verify error was displayed
                mock_error.assert_called_once()
                assert "Extraction error" in mock_error.call_args[0][0]
