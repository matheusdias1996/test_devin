"""Module for extracting entities from text using LLMs."""
from typing import Dict, Any

def _get_google_genai():
    """
    Dynamically import and return the google.generativeai module to avoid direct import.
    """
    import importlib
    return importlib.import_module("google.generativeai")


class EntityExtractor:
    """Class for extracting entities from text using LLMs."""
    
    def __init__(self, api_key: str):
        """
        Initialize the entity extractor.
        
        Args:
            api_key: Google API key
        """
        genai = _get_google_genai()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
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
                    'temperature': 0.0,
                    'top_p': 1,
                    'top_k': 1,
                }
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
            if not cleaned_response.startswith('{'):
                print("\n=== Invalid JSON format detected ===")
                # Try to find JSON in the response
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    cleaned_response = cleaned_response[start_idx:end_idx]
                    print("\n=== Extracted JSON ===")
                    print(cleaned_response)
            
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

Text to analyze:
{text}

Remember: Return ONLY the JSON object, nothing else."""
        
        return prompt 