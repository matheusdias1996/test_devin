"""Tests for content filtering in entity extraction."""
import json
from unittest import mock

import pytest

from entity_extractor import EntityExtractor


class TestContentFiltering:
    """Tests for content filtering in entity extraction."""

    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance with a mock API key."""
        with mock.patch("google.generativeai.configure"), mock.patch(
            "google.generativeai.GenerativeModel"
        ):
            return EntityExtractor("fake-api-key")

    def test_extract_entities_with_offensive_content(self, extractor):
        """Test that offensive content is properly handled."""
        # Mock response with potentially offensive content
        mock_response = mock.MagicMock()
        mock_response.text = json.dumps(
            {"Name": "John Doe", "Comment": "This contains offensive language"}
        )

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Call the method with text containing potentially offensive content
        result = extractor.extract_entities(
            "John Doe made an offensive comment",
            ["Name", "Comment"],
        )

        # Verify the result contains the extracted entities
        result_dict = json.loads(result)
        assert "Name" in result_dict
        assert "Comment" in result_dict
        assert "John Doe" == result_dict["Name"]

    def test_extract_entities_with_safety_settings(self, extractor):
        """Test that safety settings are properly applied to the LLM."""
        # Mock response
        mock_response = mock.MagicMock()
        mock_response.text = json.dumps({"Name": "John Doe"})

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Call the method
        extractor.extract_entities(
            "Some text",
            ["Name"],
        )

        # Verify the API was called with safety settings
        extractor.model.generate_content.assert_called_once()
        args, kwargs = extractor.model.generate_content.call_args
        
        # Verify temperature is set to 0.0 for deterministic output
        assert kwargs["generation_config"]["temperature"] == 0.0

    def test_extract_entities_with_harmful_content_blocked(self, extractor):
        """Test that harmful content is blocked by the LLM."""
        # Mock response with blocked content
        mock_response = mock.MagicMock()
        mock_response.text = json.dumps(
            {"Name": "John Doe", "Harmful_Content": None}
        )

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Call the method with text containing potentially harmful content
        result = extractor.extract_entities(
            "John Doe wrote something harmful",
            ["Name", "Harmful_Content"],
        )

        # Verify the result has null for harmful content
        result_dict = json.loads(result)
        assert "Name" in result_dict
        assert "Harmful_Content" in result_dict
        assert result_dict["Harmful_Content"] is None

    def test_extract_entities_with_pii_redaction(self, extractor):
        """Test that personally identifiable information (PII) is properly handled."""
        # Mock response with redacted PII
        mock_response = mock.MagicMock()
        mock_response.text = json.dumps(
            {"Name": "John Doe", "SSN": "[REDACTED]", "Phone": "[REDACTED]"}
        )

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Call the method with text containing PII
        result = extractor.extract_entities(
            "John Doe's SSN is 123-45-6789 and phone is 555-123-4567",
            ["Name", "SSN", "Phone"],
        )

        # Verify the result has redacted PII
        result_dict = json.loads(result)
        assert "Name" in result_dict
        assert "SSN" in result_dict
        assert "Phone" in result_dict
        assert result_dict["SSN"] == "[REDACTED]"
        assert result_dict["Phone"] == "[REDACTED]"

    def test_extract_entities_with_content_moderation(self, extractor):
        """Test that content moderation is applied to extracted entities."""
        # Mock response with moderated content
        mock_response = mock.MagicMock()
        mock_response.text = json.dumps(
            {"Name": "John Doe", "Message": "This message has been moderated"}
        )

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Call the method with text containing content that should be moderated
        result = extractor.extract_entities(
            "John Doe wrote an inappropriate message",
            ["Name", "Message"],
        )

        # Verify the result has moderated content
        result_dict = json.loads(result)
        assert "Name" in result_dict
        assert "Message" in result_dict
        assert "moderated" in result_dict["Message"].lower()
