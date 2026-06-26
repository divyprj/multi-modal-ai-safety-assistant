"""Configuration constants for the Multi-Modal AI Assistant."""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
SIMULATED_IMAGE_PATH = ASSETS_DIR / "simulated_scenario.png"

# Model configuration
BLIP_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
BLIP_VQA_MODEL = "Salesforce/blip-vqa-base"
GEMINI_MODEL = "gemini-2.5-flash"

# Simulated scenario metadata
SIMULATED_SCENARIO = {
    "description": "A severely frayed electrical cord lying next to a large puddle of water on a concrete floor.",
    "query": "What is the primary danger shown in this image?",
    "expected_hazard": "electrocution",
}

# Safety reasoning configuration
HAZARD_PATTERNS = {
    "electrocution": {
        "triggers": [
            (["electric", "electrical", "cord", "cable", "wire", "plug", "power", "outlet"], ["water", "wet", "puddle", "liquid", "flood", "rain", "moisture", "damp"]),
        ],
        "severity": "CRITICAL",
        "icon": "⚡",
        "description": "Electrocution Hazard",
    },
    "electrical_fire": {
        "triggers": [
            (["electric", "electrical", "cord", "cable", "wire"], ["frayed", "damaged", "broken", "exposed", "worn", "torn", "bare"]),
        ],
        "severity": "HIGH",
        "icon": "🔥",
        "description": "Electrical Fire Hazard",
    },
    "electric_shock": {
        "triggers": [
            (["electric", "electrical", "cord", "cable", "wire", "exposed"], ["touch", "contact", "handle", "grab"]),
        ],
        "severity": "HIGH",
        "icon": "⚠️",
        "description": "Electric Shock Hazard",
    },
    "slip_fall": {
        "triggers": [
            (["water", "wet", "puddle", "liquid", "spill"], ["floor", "ground", "surface", "concrete"]),
        ],
        "severity": "MEDIUM",
        "icon": "🚶",
        "description": "Slip and Fall Hazard",
    },
    "trip_hazard": {
        "triggers": [
            (["cord", "cable", "wire", "rope"], ["floor", "ground", "path", "walkway"]),
        ],
        "severity": "LOW",
        "icon": "🦶",
        "description": "Trip Hazard",
    },
}

# Severity levels with colors for UI
SEVERITY_CONFIG = {
    "CRITICAL": {"color": "#FF0040", "bg": "#2D0011", "label": "🔴 CRITICAL", "priority": 4},
    "HIGH": {"color": "#FF6B00", "bg": "#2D1A00", "label": "🟠 HIGH", "priority": 3},
    "MEDIUM": {"color": "#FFD600", "bg": "#2D2A00", "label": "🟡 MEDIUM", "priority": 2},
    "LOW": {"color": "#00E676", "bg": "#002D15", "label": "🟢 LOW", "priority": 1},
}

# Diagnostic VQA questions for scene analysis
DIAGNOSTIC_QUESTIONS = [
    "What objects are visible in this image?",
    "Is there any water or liquid in this image?",
    "Is there any electrical equipment or cord in this image?",
    "What is the condition of the objects in this image?",
    "Is there any damage visible in this image?",
    "What is on the floor?",
    "Is this scene dangerous?",
    "What material is the floor made of?",
    "Is there any hazard, danger, or unsafe condition in this image?",
    "What is the specific dangerous object or situation?",
    "What is the severity of this danger: low, medium, high, or critical?",
    "What is the immediate recommended action?",
]

# Gemini system prompt for safety analysis
GEMINI_SAFETY_PROMPT = """You are a safety analysis AI assistant. Analyze the provided image and answer the user's question with a focus on identifying potential hazards and dangers.

Your response MUST follow this exact structure:

**HAZARD IDENTIFICATION**
- Primary Hazard: [Name the specific hazard]
- Severity: [CRITICAL / HIGH / MEDIUM / LOW]
- Hazard Type: [e.g., Electrocution, Fire, Chemical, etc.]

**DETAILED ANALYSIS**
[Explain WHY this is dangerous, what specific elements in the image create the hazard, and the potential consequences]

**IMMEDIATE ACTIONS REQUIRED**
1. [First action]
2. [Second action]
3. [Third action]

**ADDITIONAL HAZARDS**
- [List any secondary hazards visible]

Be specific, technical, and actionable. Do not be generic. Reference specific objects visible in the image."""
