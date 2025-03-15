"""Extended tests for the entity extractor module."""
import json
from unittest import mock

import pytest

from entity_extractor import EntityExtractor, _get_google_genai


class TestEntityExtractorExtended:
    """Extended tests for EntityExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create an EntityExtractor instance with a mock API key."""
        with mock.patch("google.generativeai.configure"), mock.patch(
            "google.generativeai.GenerativeModel"
        ):
            return EntityExtractor("fake-api-key")

    def test_get_google_genai(self):
        """Test the dynamic import of google.generativeai."""
        with mock.patch("importlib.import_module") as mock_import:
            _get_google_genai()
            mock_import.assert_called_once_with("google.generativeai")

    def test_extract_entities_with_invalid_json_response(self, extractor):
        """Test extracting entities when LLM returns invalid JSON."""
        # Mock response with invalid JSON
        mock_response = mock.MagicMock()
        mock_response.text = "This is not JSON but contains { \"Name\": \"John Doe\" } somewhere"

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Call the method
        result = extractor.extract_entities(
            "John Doe lives at 123 Main St",
            ["Name"],
        )

        # Verify the result contains the extracted JSON
        assert "Name" in result
        assert "John Doe" in result

    def test_extract_entities_with_empty_response(self, extractor):
        """Test extracting entities when LLM returns empty response."""
        # Mock response with empty string
        mock_response = mock.MagicMock()
        mock_response.text = ""

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Call the method
        result = extractor.extract_entities(
            "Some text",
            ["Name"],
        )

        # Verify the result
        assert result == ""

    def test_extract_entities_with_api_error(self, extractor):
        """Test handling of API errors during entity extraction."""
        # Mock the generate_content method to raise an exception
        extractor.model.generate_content.side_effect = Exception("API Error")

        # Call the method and expect an exception
        with pytest.raises(Exception) as excinfo:
            extractor.extract_entities(
                "Some text",
                ["Name"],
            )

        # Verify the exception message
        assert "Error calling LLM" in str(excinfo.value)
        assert "API Error" in str(excinfo.value)

    def test_extract_entities_with_empty_entity_types(self, extractor):
        """Test extracting entities with empty entity types list."""
        # Mock response
        mock_response = mock.MagicMock()
        mock_response.text = "{}"

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Call the method
        result = extractor.extract_entities(
            "Some text",
            [],
        )

        # Verify the result
        assert result == "{}"

        # Verify the API was called with correct parameters
        extractor.model.generate_content.assert_called_once()
        args, kwargs = extractor.model.generate_content.call_args
        assert "entities:" in args[0].lower()

    def test_extract_entities_with_special_characters(self, extractor):
        """Test extracting entities with text containing special characters."""
        # Mock response
        mock_response = mock.MagicMock()
        mock_response.text = json.dumps(
            {"Email": "user@example.com", "URL": "https://example.com"}
        )

        # Mock the generate_content method
        extractor.model.generate_content.return_value = mock_response

        # Text with special characters
        text = "Contact us at user@example.com or visit https://example.com"
        
        result = extractor.extract_entities(
            text,
            ["Email", "URL"],
        )

        # Verify the result
        expected = json.dumps(
            {"Email": "user@example.com", "URL": "https://example.com"}
        )
        assert result == expected
        
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
        assert "safety_settings" in kwargs
        assert extractor.safety_settings == kwargs["safety_settings"]
        
    def test_extract_entities_with_custom_safety_settings(self):
        """Test that custom safety settings can be provided to the extractor."""
        # Skip the actual extraction and just verify safety settings are passed
        with mock.patch("entity_extractor.EntityExtractor.extract_entities") as mock_extract:
            # Create custom safety settings
            custom_safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH"
                }
            ]
            
            # Create extractor with custom safety settings
            with mock.patch("google.generativeai.configure"), mock.patch(
                "google.generativeai.GenerativeModel"
            ) as mock_model_class:
                extractor = EntityExtractor("fake-api-key", safety_settings=custom_safety_settings)
                
                # Verify the custom safety settings were used
                assert extractor.safety_settings == custom_safety_settings
                
                # Verify the model was created with the right parameters
                mock_model_class.assert_called_once_with("gemini-1.5-flash")
                
                # Verify that safety settings would be passed to generate_content
                # by checking the extractor's attributes directly
                assert hasattr(extractor, 'safety_settings')
                assert extractor.safety_settings == custom_safety_settings
