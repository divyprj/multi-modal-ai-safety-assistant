# 🛡️ Multi-Modal AI Safety Assistant

A prototype multi-modal AI assistant that processes both text and images to identify hazards and answer ambiguous safety queries. Built as a polished Streamlit web application with dual backend support.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This prototype demonstrates how a multi-modal AI system can go beyond simple image captioning to perform **contextual hazard analysis**. Given an ambiguous query like *"What is the primary danger shown in this image?"*, the system:

1. **Analyzes** the image using computer vision models
2. **Reasons** about the contextual relationships between objects
3. **Identifies** specific hazards (e.g., electrocution risk from electrical cords near water)
4. **Generates** structured safety warnings with severity ratings and recommended actions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Streamlit Web UI                    │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Image    │  │ Query    │  │ Results Panel │  │
│  │ Upload   │  │ Input    │  │ & Hazard Card │  │
│  └────┬─────┘  └────┬─────┘  └───────▲───────┘  │
│       │              │                │          │
│  ┌────▼──────────────▼────────────────┤          │
│  │        Processing Pipeline         │          │
│  │  ┌─────────────┐  ┌────────────┐   │          │
│  │  │ HF Backend  │  │ Gemini API │   │          │
│  │  │ (BLIP Cap + │  │ (Native    │   │          │
│  │  │  BLIP VQA + │  │ Multimodal)│   │          │
│  │  │  Safety     │  │            │   │          │
│  │  │  Reasoner)  │  │            │   │          │
│  │  └─────────────┘  └────────────┘   │          │
│  └────────────────────────────────────┘          │
└─────────────────────────────────────────────────┘
```

### Backends

| Feature | 🤗 HuggingFace (Local) | ✨ Google Gemini |
|---------|----------------------|-----------------|
| API Key Required | ❌ No | ✅ Yes (free) |
| Internet Required | First run only | ✅ Yes |
| Model Size | ~1.5 GB | Cloud-based |
| Answer Quality | Good (rule-enhanced) | Excellent |
| Speed | 5-15s (CPU) | 2-5s |
| Privacy | Fully local | Cloud processing |

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- 8 GB RAM minimum (for local HF backend)

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Multi-modal AI Assistant Prototype"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set up Gemini API key:**
   ```bash
   # Copy the template
   cp .env.example .env
   # Edit .env and add your API key from https://aistudio.google.com/
   ```

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🎬 Demo: Simulated Scenario

The application comes with a pre-configured safety scenario:

| Component | Details |
|-----------|---------|
| **Image** | Severely frayed electrical cord next to a large water puddle on concrete |
| **Query** | "What is the primary danger shown in this image?" |
| **Expected Output** | ⚡ **Electrocution Hazard** — CRITICAL severity |

### Steps to run the demo:

1. Launch the app with `streamlit run app.py`
2. Click **"🔄 Load Simulated Scenario"** in the sidebar
3. The query will auto-populate
4. Click **"🔍 Analyze Image"**
5. Review the hazard analysis results

## 📁 Project Structure

```
Multi-modal AI Assistant Prototype/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env.example                # Environment variable template
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── assets/
│   └── simulated_scenario.png  # Pre-generated demo image
├── core/
│   ├── __init__.py
│   ├── image_analyzer.py       # BLIP image captioning & VQA
│   ├── safety_reasoner.py      # Contextual hazard reasoning engine
│   ├── gemini_backend.py       # Google Gemini API integration
│   └── pipeline.py             # Unified processing pipeline
└── utils/
    ├── __init__.py
    └── helpers.py              # Image processing utilities
```

## 🧠 How It Works

### Local (Hugging Face) Backend

1. **Image Captioning** — `Salesforce/blip-image-captioning-large` generates a descriptive caption
2. **Visual QA** — `Salesforce/blip-vqa-base` answers 8+ diagnostic questions about the scene
3. **Safety Reasoning** — A rule-based engine analyzes the combined outputs:
   - Extracts keywords from captions and VQA answers
   - Matches against a hazard pattern database (electrical + water → electrocution)
   - Calculates confidence scores based on keyword density
   - Generates structured warnings with severity ratings

### Gemini API Backend

1. **Multimodal Input** — Image + query + safety analysis system prompt sent to Gemini
2. **Native Reasoning** — Gemini performs contextual analysis in a single pass
3. **Structured Output** — Response follows a hazard report template

## 🔧 Configuration

Key settings can be modified in `config.py`:
- Model names and versions
- Hazard patterns and severity levels
- Diagnostic VQA questions
- Gemini system prompt

## 📝 License

MIT License — Free to use and modify.
