# Multi-Modal AI Safety Assistant

A prototype multi-modal AI assistant that processes both text and images to identify hazards and answer ambiguous safety queries. Built as a modular Python application with a Streamlit web user interface and dual backend support (Hugging Face local inference and Google Gemini API).

## Overview

This prototype demonstrates how a multi-modal AI system can go beyond simple image captioning to perform contextual hazard analysis. Given an ambiguous query like "What is the primary danger shown in this image?", the system:

1. Analyzes the image using computer vision models.
2. Reasons about the contextual relationships between objects.
3. Identifies specific hazards (such as electrocution risk from electrical cords near water).
4. Generates structured safety warnings with severity ratings and recommended actions.

## Architecture

The system is built with a decoupled architecture separating the user interface, the unified pipeline, and the processing backends:

- Streamlit Web UI: A clean, dark-themed dashboard containing an image upload widget, query inputs, processing logs, and interactive hazard cards.
- Unified Pipeline: Routes processing requests to the selected backend and manages execution steps.
- Local Hugging Face Backend: 
  - Image Captioning: Uses Salesforce/blip-image-captioning-large to describe the scene.
  - Visual QA (VQA): Uses Salesforce/blip-vqa-base to query the image with specific safety-related questions.
  - Safety Reasoning Engine: A rule-based keyword matching engine that parses VQA answers and infers contextual dangers.
- Google Gemini API Backend: Leverages gemini-2.5-flash for native multimodal reasoning.

## Installation

### Prerequisites
- Python 3.10 to 3.14
- pip package manager
- 8 GB RAM minimum (for running local Hugging Face models)

### Setup Steps

1. Clone or navigate to the project directory:
   ```bash
   cd "Multi-modal AI Assistant Prototype"
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables (optional for Gemini):
   - Rename .env.example to .env
   - Open the .env file and insert your API key from Google AI Studio.

## Running the Application

To start the Streamlit web server, run:
```bash
streamlit run app.py
```
The application will be accessible at: http://localhost:8501

## Testing the Simulated Scenario

The prototype is pre-configured with a default test scenario to verify correct behavior:

- Simulated Image: A frayed electrical cord lying next to a large puddle of water on a concrete floor.
- Ambiguous Query: "What is the primary danger shown in this image?"
- Expected Inference: Electrocution Hazard (CRITICAL severity).

### How to run the test:
1. Open the web interface at http://localhost:8501.
2. Click the "Load Simulated Scenario" button in the sidebar.
3. The demo image and safety query will be loaded automatically.
4. Click the "Analyze Image" button.
5. Review the resulting hazard card and detailed explanation on the right.

## Project Structure

```
Multi-modal AI Assistant Prototype/
├── app.py                      # Main Streamlit web application
├── config.py                   # Configuration parameters and hazard patterns
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore file
├── .streamlit/
│   └── config.toml             # Theme settings for Streamlit UI
├── assets/
│   └── simulated_scenario.png  # Pre-generated scenario image for demo
├── core/
│   ├── __init__.py
│   ├── image_analyzer.py       # Local Hugging Face BLIP and VQA loader
│   ├── safety_reasoner.py      # Keyword processing and safety engine
│   ├── gemini_backend.py       # Google Gemini API connector
│   └── pipeline.py             # Pipeline orchestrator
└── utils/
    ├── __init__.py
    └── helpers.py              # Image and time formatting helper functions
```

## License

This project is licensed under the MIT License.
