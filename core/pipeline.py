"""Unified processing pipeline for multi-modal analysis.

Orchestrates the image analysis, safety reasoning, and API backends
into a single processing flow.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List
from PIL import Image
from core.image_analyzer import ImageAnalyzer, SceneAnalysis
from core.safety_reasoner import SafetyReasoner, SafetyReport
from core.gemini_backend import GeminiAnalyzer, GeminiResult


@dataclass
class PipelineStep:
    """A single step in the processing pipeline."""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    duration: float = 0.0
    detail: str = ""


@dataclass
class AnalysisResult:
    """Complete result from the processing pipeline."""
    backend: str = ""  # "huggingface" or "gemini"
    scene_analysis: Optional[SceneAnalysis] = None
    safety_report: Optional[SafetyReport] = None
    gemini_result: Optional[GeminiResult] = None
    total_time: float = 0.0
    steps: List[PipelineStep] = field(default_factory=list)
    success: bool = False
    error: str = ""


class MultiModalPipeline:
    """Unified processing pipeline for multi-modal analysis.
    
    Routes processing to either the local Hugging Face backend
    or the Google Gemini API backend.
    """

    def __init__(self):
        self._hf_analyzer = None
        self._safety_reasoner = SafetyReasoner()

    def _get_hf_analyzer(self) -> ImageAnalyzer:
        """Get or create the HuggingFace analyzer (lazy loading)."""
        if self._hf_analyzer is None:
            self._hf_analyzer = ImageAnalyzer()
        return self._hf_analyzer

    def process_with_huggingface(
        self, image: Image.Image, query: str, progress_callback=None
    ) -> AnalysisResult:
        """Process using local Hugging Face BLIP models.
        
        Args:
            image: PIL Image to analyze.
            query: User's question about the image.
            progress_callback: Optional callback(step_name, status) for UI updates.
            
        Returns:
            AnalysisResult with scene analysis and safety report.
        """
        result = AnalysisResult(backend="huggingface")
        start_time = time.time()
        
        try:
            # Step 1: Load models
            step = PipelineStep(name="Loading BLIP Models")
            if progress_callback:
                progress_callback(step.name, "running")
            
            t0 = time.time()
            analyzer = self._get_hf_analyzer()
            step.duration = time.time() - t0
            step.status = "completed"
            result.steps.append(step)
            
            if progress_callback:
                progress_callback(step.name, "completed")
            
            # Step 2: Scene analysis (captioning + VQA)
            step = PipelineStep(name="Analyzing Scene")
            if progress_callback:
                progress_callback(step.name, "running")
            
            t0 = time.time()
            additional_q = [query] if query not in ["", None] else []
            scene = analyzer.analyze_scene(image, additional_questions=additional_q)
            result.scene_analysis = scene
            step.duration = time.time() - t0
            step.status = "completed"
            step.detail = f"Caption: {scene.caption}"
            result.steps.append(step)
            
            if progress_callback:
                progress_callback(step.name, "completed")
            
            # Step 3: Safety reasoning
            step = PipelineStep(name="Safety Reasoning")
            if progress_callback:
                progress_callback(step.name, "running")
            
            t0 = time.time()
            report = self._safety_reasoner.analyze(
                caption=scene.caption,
                vqa_answers=scene.vqa_answers,
                raw_texts=scene.raw_texts,
            )
            result.safety_report = report
            step.duration = time.time() - t0
            step.status = "completed"
            if report.primary_hazard:
                step.detail = f"Primary: {report.primary_hazard.hazard_name}"
            result.steps.append(step)
            
            if progress_callback:
                progress_callback(step.name, "completed")
            
            result.success = True
            
        except Exception as e:
            result.error = str(e)
            result.success = False
        
        result.total_time = time.time() - start_time
        return result

    def process_with_gemini(
        self, image: Image.Image, query: str, api_key: str, progress_callback=None
    ) -> AnalysisResult:
        """Process using Google Gemini API.
        
        Args:
            image: PIL Image to analyze.
            query: User's question about the image.
            api_key: Gemini API key.
            progress_callback: Optional callback for UI updates.
            
        Returns:
            AnalysisResult with Gemini response.
        """
        result = AnalysisResult(backend="gemini")
        start_time = time.time()
        
        try:
            # Step 1: Initialize Gemini
            step = PipelineStep(name="Connecting to Gemini API")
            if progress_callback:
                progress_callback(step.name, "running")
            
            t0 = time.time()
            gemini = GeminiAnalyzer(api_key=api_key)
            step.duration = time.time() - t0
            step.status = "completed"
            result.steps.append(step)
            
            if progress_callback:
                progress_callback(step.name, "completed")
            
            # Step 2: Analyze with Gemini
            step = PipelineStep(name="Gemini Multimodal Analysis")
            if progress_callback:
                progress_callback(step.name, "running")
            
            t0 = time.time()
            gemini_result = gemini.analyze(image, query)
            result.gemini_result = gemini_result
            step.duration = time.time() - t0
            step.status = "completed" if gemini_result.success else "failed"
            step.detail = gemini_result.error if not gemini_result.success else "Analysis complete"
            result.steps.append(step)
            
            if progress_callback:
                progress_callback(step.name, step.status)
            
            result.success = gemini_result.success
            if not gemini_result.success:
                result.error = gemini_result.error
            
        except Exception as e:
            result.error = str(e)
            result.success = False
        
        result.total_time = time.time() - start_time
        return result

    def process(
        self,
        image: Image.Image,
        query: str,
        backend: str = "huggingface",
        gemini_api_key: str = None,
        progress_callback=None,
    ) -> AnalysisResult:
        """Process an image with a query using the specified backend.
        
        Args:
            image: PIL Image to analyze.
            query: User's question about the image.
            backend: 'huggingface' or 'gemini'.
            gemini_api_key: Required if backend is 'gemini'.
            progress_callback: Optional callback for UI updates.
            
        Returns:
            AnalysisResult from the selected backend.
        """
        if backend == "gemini":
            if not gemini_api_key:
                result = AnalysisResult(backend="gemini")
                result.error = "Gemini API key is required. Enter it in the sidebar."
                return result
            return self.process_with_gemini(
                image, query, gemini_api_key, progress_callback
            )
        else:
            return self.process_with_huggingface(
                image, query, progress_callback
            )
