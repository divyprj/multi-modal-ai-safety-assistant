# 🛡️ Multi-Modal AI Safety Assistant

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Backend](https://img.shields.io/badge/backend-transformers%20%7C%20gemini-orange.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

A prototype multi-modal artificial intelligence safety assistant that processes both text and images to identify hazards and answer ambiguous safety queries. Built as a modular Python application with a Streamlit web user interface and dual backend support (Hugging Face local inference and Google Gemini API).

---

## 🎯 Overview

This prototype demonstrates how a multi-modal AI system can go beyond simple image captioning to perform **contextual hazard analysis**. Given an ambiguous query like *"What is the primary danger shown in this image?"*, the system:

1. 🔍 **Analyzes** the image using computer vision models.
2. 🧠 **Reasons** about the contextual relationships between objects.
3. ⚡ **Identifies** specific hazards (such as electrocution risk from electrical cords near water).
4. 📊 **Generates** structured safety warnings with severity ratings and recommended actions.

---

## 🚀 Core Features

* 🔧 **Dual-Backend Support**: Toggle between local Hugging Face model inference (offline execution) and the Google Gemini API (high-accuracy cloud reasoning).
* 🧠 **Contextual Safety Reasoning**: Evaluates relationships between multiple visual objects (like electrical cables near water puddles) to deduce safety hazards that are not explicitly stated.
* 🚨 **Structured Risk Reporting**: Automatically generates detailed hazard summaries including severity levels (Critical, High, Medium, Low), hazard categories, detailed risk explanations, and immediate safety recommendations.
* 💻 **Interactive UI**: A polished, dark-themed dashboard built with Streamlit featuring responsive columns, image preview widgets, and custom CSS status cards.

---

## 🏗️ Architecture Overview

The application follows a decoupled model-view-controller design:

* 🖥️ **View (app.py)**: Streamlit-based web dashboard managing file upload, text input, configuration selectors, and HTML-injected warning blocks.
* ⚙️ **Controller (core/pipeline.py)**: Orchestrates image preprocessing, backend routing, step-by-step pipeline logging, and error handling.
* 🧠 **Model Engine (core/image_analyzer.py & core/safety_reasoner.py)**: Loads local model weights, queries visual question-answering pipelines, and performs rule-based and dynamic keyword-association mapping to identify hazards.
* 🌐 **Cloud API (core/gemini_backend.py)**: Standardizes payload requests to the Google GenAI SDK and parses the response markdown.

---

## 📊 Backend Comparison

| Parameter | 🤗 Local Hugging Face Backend | ✨ Google Gemini API Backend |
| :--- | :--- | :--- |
| **Model Weights** | Salesforce/blip-image-captioning-large (~1.88 GB)<br>Salesforce/blip-vqa-base (~1.54 GB) | Hosted API (gemini-2.5-flash) |
| **Inference Location** | Local machine CPU/GPU | Google Cloud server |
| **Key Prerequisites** | 8 GB RAM minimum | AI Studio API key (free tier available) |
| **Accuracy Profile** | High for configured rules (rules-guided VQA) | Excellent across arbitrary unseen hazards |
| **Network Dependency** | First run only (for downloading weights) | Constant internet connection required |

---

## 📦 Setup and Installation

### 📋 Prerequisites
* Python 3.10, 3.11, or 3.12
* pip (Python package installer)

### 🛠️ Installation Steps

1. Clone or navigate to the directory:
   ```bash
   cd "Multi-modal AI Assistant Prototype"
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   * **Windows Command Prompt:**
     ```cmd
     venv\Scripts\activate.bat
     ```
   * **Windows PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **macOS/Linux Terminal:**
     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. 🔑 Configure the Gemini API key (Optional):
   * Duplicate the environment template file:
     ```bash
     copy .env.example .env
     ```
   * Open the `.env` file and insert your API key from Google AI Studio.

---

## 🎥 Explanation and Demo Video

Watch the detailed walkthrough of the prototype, showing setup, architecture, and live hazard detection:
👉 **[Watch the Demo Video on YouTube](https://youtu.be/6KHJ36AFkE4)**

---

## 🎮 Running the Application

Execute the Streamlit application command:
```bash
streamlit run app.py
```
After initialization, the web interface will load at: http://localhost:8501

### 🎬 Walkthrough of the Pre-configured Safety Demo
To verify the system's contextual hazard deduction:
1. Open the UI at http://localhost:8501.
2. Click **🔄 Load Simulated Scenario** on the sidebar to load the test image (a frayed power cable next to a water spill).
3. The test query *"What is the primary danger shown in this image?"* will be pre-filled.
4. Click **🔍 Analyze Image**.
5. The local pipeline will run the BLIP image analyzer, query the VQA module, and present a **Critical Electrocution Hazard** card along with 7 recommended safety actions.

---

## 📁 File Structure

```
Multi-modal AI Assistant Prototype/
├── app.py                      # Main Streamlit web interface
├── config.py                   # Setup constants and hazard pattern definitions
├── requirements.txt            # Python dependencies configuration
├── README.md                   # Project documentation
├── .env.example                # Template for API credentials
├── .gitignore                  # Excluded git tracking patterns
├── .streamlit/
│   └── config.toml             # Streamlit visual customization
├── assets/
│   └── simulated_scenario.png  # Pre-generated scenario image for demo
├── core/
│   ├── __init__.py
│   ├── image_analyzer.py       # Handles local BLIP models
│   ├── safety_reasoner.py      # Logic for keyword mapping and safety analysis
│   ├── gemini_backend.py       # Integration for Google Gemini API
│   └── pipeline.py             # Orchestrates the run steps
└── utils/
    ├── __init__.py
    └── helpers.py              # Visual helpers and timing utilities
```

## 📝 License

This project is licensed under the MIT License.
