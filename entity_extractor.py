"""Module for extracting entities from text using LLMs."""
from typing import Dict, Any, List, Optional


def _get_google_genai():
    """
    Dynamically import and return the google.generativeai module to avoid direct import.
    """
    import importlib

    genai = importlib.import_module("google.generativeai")
    return genai

# Default safety settings to block offensive content
DEFAULT_SAFETY_SETTINGS_V2 = None  # Will be initialized in the class


class EntityExtractor:
    """Class for extracting entities from text using LLMs."""

    def __init__(self, api_key: str, safety_settings: Optional[List[Dict]] = None):
        """
        Initialize the entity extractor.

        Args:
            api_key: Google API key
            safety_settings: Optional safety settings for content filtering
        """
        genai = _get_google_genai()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.safety_settings = safety_settings
        
        # Initialize default safety settings if not provided
        if self.safety_settings is None:
            HarmCategory = genai.types.HarmCategory
            HarmBlockThreshold = genai.types.HarmBlockThreshold
            
            self.safety_settings = [
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                }
            ]

    def extract_entities(self, text: str, entity_types: list[str]) -> Dict[str, Any]:
        """
        Extract specified entities from text.

        Args:
            text: Text to extract entities from
            entity_types: List of entity types to extract

        Returns:
            Dictionary of extracted entities
        """
        # Create prompt for the LLM
        prompt = self._create_extraction_prompt(text, entity_types)
        print("\n=== Prompt sent to LLM ===")
        print(prompt)

        try:
            # Call Gemini with safety settings
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "top_p": 1,
                    "top_k": 1,
                },
                safety_settings=self.safety_settings
            )

            # Print raw response for debugging
            print("\n=== Raw LLM Response ===")
            print(response.text)
            print("\n=== Response Type ===")
            print(type(response.text))

            # Try to clean and parse the response
            cleaned_response = response.text.strip()
            print("\n=== Cleaned Response ===")
            print(cleaned_response)

            # Ensure we have a valid JSON string
            if not cleaned_response.startswith("{"):
                print("\n=== Invalid JSON format detected ===")
                # Try to find JSON in the response
                start_idx = cleaned_response.find("{")
                end_idx = cleaned_response.rfind("}") + 1
                if start_idx != -1 and end_idx != 0:
                    cleaned_response = cleaned_response[start_idx:end_idx]
                    print("\n=== Extracted JSON ===")
                    print(cleaned_response)
            
            # Post-process for content filtering
            try:
                import json
                result_dict = json.loads(cleaned_response)
                
                # Check for potentially offensive content and redact if necessary
                for key, value in result_dict.items():
                    if isinstance(value, str) and any(term in value.lower() for term in ["offensive", "explicit", "harmful", "inappropriate"]):
                        result_dict[key] = "[CONTENT_REMOVED]"
                    
                    # Redact common PII patterns
                    if isinstance(value, str) and any(pattern in value.lower() for pattern in ["ssn", "social security", "credit card", "password"]):
                        result_dict[key] = "[REDACTED]"
                
                cleaned_response = json.dumps(result_dict)
            except (json.JSONDecodeError, AttributeError):
                pass  # If not valid JSON, skip post-processing

            return cleaned_response

        except Exception as e:
            print("\n=== Error Details ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise Exception(f"Error calling LLM: {str(e)}")

    def _create_extraction_prompt(self, text: str, entity_types: list[str]) -> str:
        """
        Create a prompt for entity extraction.

        Args:
            text: Text to extract entities from
            entity_types: List of entity types to extract

        Returns:
            Prompt for the LLM
        """
        entities_str = ", ".join(entity_types)

        prompt = f"""You are a JSON generator. Your task is to extract specific entities from text and format them as JSON.

Extract these entities: {entities_str}

Rules:
1. Output must be ONLY a valid JSON object
2. Keys must be exactly as specified
3. Values should be the extracted entities or null if not found
4. Do not include any explanations or additional text
5. Do not include any offensive, harmful, or inappropriate content
6. Redact any personally identifiable information (PII) with [REDACTED]
7. If offensive content is detected, replace it with [CONTENT_REMOVED]

Text to analyze:
{text}

Remember: Return ONLY the JSON object, nothing else."""

        return prompt
