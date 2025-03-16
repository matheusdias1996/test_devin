"""Tests for the summarization functionality."""
import pytest
from unittest import mock

from entity_extractor import EntityExtractor


class TestSummarization:
    """Tests for summarization functionality."""

    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance with a mock API key."""
        with mock.patch("google.generativeai.configure"), mock.patch(
            "google.generativeai.GenerativeModel"
        ):
            return EntityExtractor("fake-api-key")

    def test_create_summary_prompt(self, extractor):
        """Test creating a summary prompt."""
        text = "Sample text for summarization"
        
        # Test without max_length
        prompt = extractor._create_summary_prompt(text)
        assert "Sample text for summarization" in prompt
        assert "concise summary" in prompt
        assert "no longer than" not in prompt
        
        # Test with max_length
        prompt = extractor._create_summary_prompt(text, max_length=100)
        assert "Sample text for summarization" in prompt
        assert "concise summary" in prompt
        assert "no longer than 100 words" in prompt

    def test_summarize_text(self, extractor):
        """Test summarizing text."""
        # Mock response
        mock_response = mock.MagicMock()
        mock_response.text = "This is a summary of the text."

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        result = extractor.summarize_text(
            "This is a long text that needs to be summarized for testing purposes.",
            max_length=50,
        )

        # Verify the result
        assert result == "This is a summary of the text."

        # Verify the API was called with correct parameters
        extractor.model.generate_content.assert_called_once()
        args, kwargs = extractor.model.generate_content.call_args
        assert "summarize" in args[0].lower()
        assert "50 words" in args[0]
        assert kwargs["generation_config"]["temperature"] == 0.2
