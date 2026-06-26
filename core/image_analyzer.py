"""Image analysis module using Hugging Face BLIP models.

Provides image captioning and visual question answering (VQA)
capabilities using the Salesforce BLIP models.
"""

import time
from dataclasses import dataclass, field
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from config import BLIP_CAPTION_MODEL, BLIP_VQA_MODEL, DIAGNOSTIC_QUESTIONS


@dataclass
class SceneAnalysis:
    """Results from a complete scene analysis."""
    caption: str = ""
    vqa_answers: dict = field(default_factory=dict)  # question -> answer
    raw_texts: list = field(default_factory=list)  # all text outputs combined
    processing_time: float = 0.0


class ImageAnalyzer:
    """BLIP-based image analysis engine.
    
    Uses Salesforce BLIP models for image captioning and
    visual question answering. Models are lazily loaded on first use.
    """

    def __init__(self):
        self._caption_processor = None
        self._caption_model = None
        self._vqa_processor = None
        self._vqa_model = None

    def _load_caption_model(self):
        """Lazily load the BLIP captioning model."""
        if self._caption_processor is None:
            self._caption_processor = BlipProcessor.from_pretrained(BLIP_CAPTION_MODEL)
            self._caption_model = BlipForConditionalGeneration.from_pretrained(BLIP_CAPTION_MODEL)
            self._caption_model.eval()

    def _load_vqa_model(self):
        """Lazily load the BLIP VQA model."""
        if self._vqa_processor is None:
            self._vqa_processor = BlipProcessor.from_pretrained(BLIP_VQA_MODEL)
            self._vqa_model = BlipForQuestionAnswering.from_pretrained(BLIP_VQA_MODEL)
            self._vqa_model.eval()

    def generate_caption(self, image: Image.Image) -> str:
        """Generate a descriptive caption for the image.
        
        Args:
            image: PIL Image to caption.
            
        Returns:
            A text caption describing the image contents.
        """
        self._load_caption_model()
        image_rgb = image.convert("RGB")
        
        # Unconditional captioning
        inputs = self._caption_processor(image_rgb, return_tensors="pt")
        out = self._caption_model.generate(**inputs, max_new_tokens=100)
        caption = self._caption_processor.decode(out[0], skip_special_tokens=True)
        
        return caption

    def generate_conditional_caption(self, image: Image.Image, prompt: str) -> str:
        """Generate a caption conditioned on a text prompt.
        
        Args:
            image: PIL Image to caption.
            prompt: Text prompt to condition the caption on.
            
        Returns:
            A conditioned text caption.
        """
        self._load_caption_model()
        image_rgb = image.convert("RGB")
        
        inputs = self._caption_processor(image_rgb, text=prompt, return_tensors="pt")
        out = self._caption_model.generate(**inputs, max_new_tokens=100)
        caption = self._caption_processor.decode(out[0], skip_special_tokens=True)
        
        return caption

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about the image using BLIP VQA.
        
        Args:
            image: PIL Image to analyze.
            question: Question about the image.
            
        Returns:
            Short text answer to the question.
        """
        self._load_vqa_model()
        image_rgb = image.convert("RGB")
        
        inputs = self._vqa_processor(image_rgb, question, return_tensors="pt")
        out = self._vqa_model.generate(**inputs, max_new_tokens=50)
        answer = self._vqa_processor.decode(out[0], skip_special_tokens=True)
        
        return answer

    def analyze_scene(self, image: Image.Image, additional_questions: list = None) -> SceneAnalysis:
        """Perform comprehensive scene analysis.
        
        Runs image captioning and multiple diagnostic VQA questions
        to build a complete understanding of the scene.
        
        Args:
            image: PIL Image to analyze.
            additional_questions: Optional extra questions to ask.
            
        Returns:
            SceneAnalysis with caption, VQA answers, and timing.
        """
        start_time = time.time()
        result = SceneAnalysis()
        
        # Step 1: Generate caption
        result.caption = self.generate_caption(image)
        result.raw_texts.append(result.caption)
        
        # Step 2: Generate conditional captions for more detail
        conditional_prompts = [
            "a photo of",
            "this image shows",
            "the danger in this image is",
        ]
        for prompt in conditional_prompts:
            cond_caption = self.generate_conditional_caption(image, prompt)
            result.raw_texts.append(cond_caption)
        
        # Step 3: Run diagnostic VQA questions
        questions = list(DIAGNOSTIC_QUESTIONS)
        if additional_questions:
            questions.extend(additional_questions)
        
        for question in questions:
            answer = self.answer_question(image, question)
            result.vqa_answers[question] = answer
            result.raw_texts.append(answer)
        
        result.processing_time = time.time() - start_time
        return result
