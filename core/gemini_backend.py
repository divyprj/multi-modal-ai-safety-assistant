"""Google Gemini API backend for multi-modal analysis.

Provides image analysis using Google's Gemini vision models
for more sophisticated contextual understanding.
"""

import io
import time
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from config import GEMINI_MODEL, GEMINI_SAFETY_PROMPT


@dataclass
class GeminiResult:
    """Result from Gemini API analysis."""
    response_text: str = ""
    model_used: str = ""
    processing_time: float = 0.0
    success: bool = False
    error: str = ""


class GeminiAnalyzer:
    """Google Gemini API-based image analyzer.
    
    Uses Google's multimodal Gemini models for sophisticated
    image understanding and safety analysis.
    """

    def __init__(self, api_key: str):
        """Initialize with a Gemini API key.
        
        Args:
            api_key: Google AI Studio API key.
        """
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package is required for Gemini backend. "
                    "Install with: pip install google-genai"
                )
        return self._client

    def analyze(self, image: Image.Image, query: str) -> GeminiResult:
        """Analyze an image with a text query using Gemini.
        
        Args:
            image: PIL Image to analyze.
            query: User's question about the image.
            
        Returns:
            GeminiResult with the model's response.
        """
        start_time = time.time()
        result = GeminiResult(model_used=GEMINI_MODEL)
        
        try:
            client = self._get_client()
            
            # Convert PIL Image to bytes
            img_buffer = io.BytesIO()
            image_rgb = image.convert("RGB")
            image_rgb.save(img_buffer, format="JPEG", quality=90)
            img_bytes = img_buffer.getvalue()
            
            # Create image part
            from google.genai.types import Part
            image_part = Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
            
            # Send to Gemini with safety analysis prompt
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    GEMINI_SAFETY_PROMPT,
                    image_part,
                    f"User's question: {query}",
                ],
            )
            
            result.response_text = response.text
            result.success = True
            
        except ImportError as e:
            result.error = str(e)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                result.error = (
                    "Gemini API rate limit reached. Please wait a moment and try again, "
                    "or switch to the Local (Hugging Face) backend."
                )
            elif "API_KEY" in error_msg or "401" in error_msg or "403" in error_msg:
                result.error = (
                    "Invalid or missing Gemini API key. Please check your API key "
                    "in the sidebar settings."
                )
            else:
                result.error = f"Gemini API error: {error_msg}"
        
        result.processing_time = time.time() - start_time
        return result

    def test_connection(self) -> bool:
        """Test if the API key is valid and the service is reachable."""
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=["Say 'OK' if you can hear me."],
            )
            return bool(response.text)
        except Exception:
            return False
