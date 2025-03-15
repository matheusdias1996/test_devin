"""Tests for the entity extractor module."""
import json
from unittest import mock

import pytest

from entity_extractor import EntityExtractor


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance with a mock API key."""
        with mock.patch("google.generativeai.configure"), \
             mock.patch("google.generativeai.GenerativeModel"):
            return EntityExtractor("fake-api-key")

    def test_create_extraction_prompt(self, extractor):
        """Test creating an extraction prompt."""
        text = "Sample text for extraction"
        entity_types = ["Name", "Date", "Address"]
        
        prompt = extractor._create_extraction_prompt(text, entity_types)
        
        assert "Name, Date, Address" in prompt
        assert "Sample text for extraction" in prompt
        assert "JSON" in prompt

    def test_extract_entities(self, extractor):
        """Test extracting entities from text."""
        # Mock response
        mock_response = mock.MagicMock()
        mock_response.text = json.dumps({
            "Name": "John Doe",
            "Date": "2023-11-15",
            "Address": "123 Main St"
        })
        
        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response
        
        result = extractor.extract_entities(
            "John Doe lives at 123 Main St since 2023-11-15",
            ["Name", "Date", "Address"]
        )
        
        # Verify the result
        expected = json.dumps({
            "Name": "John Doe",
            "Date": "2023-11-15",
            "Address": "123 Main St"
        })
        
        assert result == expected
        
        # Verify the API was called with correct parameters
        extractor.model.generate_content.assert_called_once()
        args, kwargs = extractor.model.generate_content.call_args
        assert "Name, Date, Address" in args[0]
        assert kwargs["generation_config"]["temperature"] == 0.0 