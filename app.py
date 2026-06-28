"""
Multi-Modal AI Assistant Prototype
===================================
A Streamlit-based interactive application that processes both text and images
to identify hazards and answer ambiguous safety queries.

Supports two backends:
- Local Hugging Face BLIP models (no API key required)
- Google Gemini API (free tier, better contextual reasoning)
"""

import streamlit as st
from PIL import Image
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SIMULATED_IMAGE_PATH,
    SIMULATED_SCENARIO,
    SEVERITY_CONFIG,
)
from utils.helpers import resize_image, format_time

# ── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multi-Modal AI Safety Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }

    .main-header h1 {
        background: linear-gradient(135deg, #6C63FF, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #94A3B8;
        font-size: 1rem;
        font-weight: 300;
    }

    /* Hazard card */
    .hazard-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
    }

    .hazard-critical {
        background: linear-gradient(135deg, rgba(255, 0, 64, 0.08), rgba(255, 0, 64, 0.03));
        border-left-color: #FF0040;
        border: 1px solid rgba(255, 0, 64, 0.2);
    }

    .hazard-high {
        background: linear-gradient(135deg, rgba(255, 107, 0, 0.08), rgba(255, 107, 0, 0.03));
        border-left-color: #FF6B00;
        border: 1px solid rgba(255, 107, 0, 0.2);
    }

    .hazard-medium {
        background: linear-gradient(135deg, rgba(255, 214, 0, 0.08), rgba(255, 214, 0, 0.03));
        border-left-color: #FFD600;
        border: 1px solid rgba(255, 214, 0, 0.2);
    }

    .hazard-low {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.08), rgba(0, 230, 118, 0.03));
        border-left-color: #00E676;
        border: 1px solid rgba(0, 230, 118, 0.2);
    }

    /* Severity badge */
    .severity-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .severity-critical { background: rgba(255, 0, 64, 0.2); color: #FF0040; }
    .severity-high { background: rgba(255, 107, 0, 0.2); color: #FF6B00; }
    .severity-medium { background: rgba(255, 214, 0, 0.2); color: #FFD600; }
    .severity-low { background: rgba(0, 230, 118, 0.2); color: #00E676; }

    /* Info card */
    .info-card {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.06), rgba(59, 130, 246, 0.03));
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
    }

    /* Pipeline step */
    .pipeline-step {
        display: flex;
        align-items: center;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .step-completed { border-left: 3px solid #00E676; }
    .step-running { border-left: 3px solid #FFD600; }
    .step-failed { border-left: 3px solid #FF0040; }

    /* Action item */
    .action-item {
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.95rem;
    }

    /* Confidence meter */
    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: rgba(255, 255, 255, 0.1);
        overflow: hidden;
        margin-top: 0.5rem;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: rgba(108, 99, 255, 0.05);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
    }

    /* Metrics row */
    .metrics-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    .metric-card {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #6C63FF;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Scene element pills */
    .scene-pill {
        display: inline-block;
        padding: 4px 12px;
        margin: 2px 4px;
        border-radius: 16px;
        background: rgba(108, 99, 255, 0.12);
        border: 1px solid rgba(108, 99, 255, 0.25);
        color: #A5B4FC;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Pulse animation for running steps */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .step-running { animation: pulse 1.5s infinite; }

    /* Divider */
    .styled-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(108, 99, 255, 0.3), transparent);
        margin: 1.5rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ────────────────────────────────────────────

if "pipeline" not in st.session_state:
    from core.pipeline import MultiModalPipeline
    st.session_state.pipeline = MultiModalPipeline()
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "loaded_image" not in st.session_state:
    st.session_state.loaded_image = None
if "image_source" not in st.session_state:
    st.session_state.image_source = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# ── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🛡️ Multi-Modal AI Safety Assistant</h1>
    <p>Advanced hazard detection powered by computer vision and contextual reasoning</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🔧 Backend Selection")
    backend = st.radio(
        "Choose analysis backend:",
        ["🤗 Local (Hugging Face BLIP)", "✨ Google Gemini API"],
        index=0,
        help="Local runs BLIP models on your machine. Gemini requires a free API key.",
    )
    backend_key = "huggingface" if "Local" in backend else "gemini"

    # Gemini API key input
    gemini_api_key = None
    if backend_key == "gemini":
        st.markdown("### 🔑 Gemini API Key")
        gemini_api_key = st.text_input(
            "Enter your API key:",
            type="password",
            help="Get a free key at https://aistudio.google.com/",
        )
        if not gemini_api_key:
            st.warning("⚠️ API key required for Gemini backend")
        st.markdown(
            "[Get free API key →](https://aistudio.google.com/)",
        )

    st.markdown("---")

    # Simulated scenario loader
    st.markdown("### 🎯 Demo Scenario")
    st.caption(
        "Load the pre-configured safety scenario to test the system's "
        "ability to identify contextual hazards."
    )
    if st.button("🔄 Load Simulated Scenario", use_container_width=True):
        if SIMULATED_IMAGE_PATH.exists():
            st.session_state.loaded_image = Image.open(SIMULATED_IMAGE_PATH)
            st.session_state.image_source = "simulated"
            st.session_state.analysis_result = None
            st.session_state.uploaded_file_name = None
            st.rerun()
        else:
            st.error("Simulated scenario image not found in assets/")

    st.markdown("---")

    # Architecture info
    st.markdown("### 🏗️ Architecture")
    if backend_key == "huggingface":
        st.markdown("""
        **Pipeline:**
        1. 📷 BLIP Image Captioning
        2. ❓ BLIP Visual QA (8 diagnostic questions)
        3. 🧠 Safety Reasoning Engine
        4. 📊 Hazard Report Generation
        """)
    else:
        st.markdown("""
        **Pipeline:**
        1. 📷 Image encoding
        2. 🤖 Gemini multimodal analysis
        3. 📊 Structured hazard report
        """)


# ── Main Content ────────────────────────────────────────────────────────────

col_input, col_output = st.columns([1, 1], gap="large")

# ── Left Column: Input ──────────────────────────────────────────────────────

with col_input:
    st.markdown("### 📷 Image Input")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image for safety analysis",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload any image to analyze for potential hazards.",
    )

    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.loaded_image = Image.open(uploaded_file)
            st.session_state.image_source = "uploaded"
            st.session_state.analysis_result = None
            st.session_state.uploaded_file_name = uploaded_file.name
    else:
        if st.session_state.image_source == "uploaded":
            st.session_state.loaded_image = None
            st.session_state.image_source = None
            st.session_state.analysis_result = None
            st.session_state.uploaded_file_name = None

    # Display the current image
    current_image = st.session_state.loaded_image
    if current_image is not None:
        # Resize for display
        display_image = resize_image(current_image, max_size=800)
        st.image(
            display_image,
            caption=(
                "🎯 Simulated Scenario"
                if st.session_state.image_source == "simulated"
                else "📷 Uploaded Image"
            ),
            use_container_width=True,
        )

        # Image metadata
        w, h = current_image.size
        st.caption(f"📐 {w} × {h} px | {current_image.mode}")
    else:
        st.markdown(
            """
            <div class="info-card">
                <p style="text-align: center; color: #94A3B8; margin: 1rem 0;">
                    📷 Upload an image above or click 
                    <strong>"Load Simulated Scenario"</strong> in the sidebar
                    to get started.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Query input
    st.markdown("### 💬 Query")

    default_query = (
        SIMULATED_SCENARIO["query"]
        if st.session_state.image_source == "simulated"
        else ""
    )

    query = st.text_area(
        "Ask a question about the image:",
        value=default_query,
        height=80,
        placeholder="e.g., What is the primary danger shown in this image?",
    )

    # Analyze button
    can_analyze = current_image is not None and query.strip() != ""
    if backend_key == "gemini" and not gemini_api_key:
        can_analyze = False

    analyze_clicked = st.button(
        "🔍 Analyze Image",
        use_container_width=True,
        disabled=not can_analyze,
        type="primary",
    )

    if not can_analyze and current_image is not None and query.strip() != "":
        st.info("💡 Enter your Gemini API key in the sidebar to continue.")


# ── Run Analysis ────────────────────────────────────────────────────────────

if analyze_clicked and can_analyze:
    with col_output:
        st.markdown("### 🔬 Analysis Results")

        # Progress display
        progress_container = st.empty()
        status_text = st.empty()

        with st.spinner("🔄 Processing..."):
            # Resize image for model input
            model_image = resize_image(current_image, max_size=512)

            # Run the pipeline
            result = st.session_state.pipeline.process(
                image=model_image,
                query=query,
                backend=backend_key,
                gemini_api_key=gemini_api_key,
            )

            st.session_state.analysis_result = result

        st.rerun()


# ── Display Results ─────────────────────────────────────────────────────────

with col_output:
    result = st.session_state.analysis_result

    if result is None:
        st.markdown("### 🔬 Analysis Results")
        st.markdown(
            """
            <div class="info-card">
                <p style="text-align: center; color: #94A3B8; margin: 1rem 0;">
                    🔍 Results will appear here after analysis.<br>
                    <small>Load an image, enter a query, and click "Analyze Image"</small>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not result.success:
        st.markdown("### 🔬 Analysis Results")
        st.error(f"❌ Analysis failed: {result.error}")
    else:
        st.markdown("### 🔬 Analysis Results")

        # ── Processing Metrics ──────────────────────────────────────────
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("⏱️ Total Time", format_time(result.total_time))
        with metric_cols[1]:
            st.metric("🔧 Backend", result.backend.upper())
        with metric_cols[2]:
            steps_done = sum(1 for s in result.steps if s.status == "completed")
            st.metric("📋 Steps", f"{steps_done}/{len(result.steps)}")

        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

        # ── Hugging Face Results ────────────────────────────────────────
        if result.backend == "huggingface" and result.safety_report:
            report = result.safety_report

            # Primary Hazard Card
            if report.primary_hazard:
                hazard = report.primary_hazard
                sev_lower = hazard.severity.lower()
                sev_config = SEVERITY_CONFIG.get(hazard.severity, {})

                st.markdown(
                    f"""
                    <div class="hazard-card hazard-{sev_lower}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h3 style="margin: 0; font-size: 1.3rem;">
                                {hazard.icon} {hazard.hazard_name}
                            </h3>
                            <span class="severity-badge severity-{sev_lower}">
                                {sev_config.get('label', hazard.severity)}
                            </span>
                        </div>
                        <div class="confidence-bar" style="margin-top: 0.8rem;">
                            <div class="confidence-fill" style="width: {hazard.confidence*100:.0f}%; background: {sev_config.get('color', '#FFF')};"></div>
                        </div>
                        <p style="color: #94A3B8; font-size: 0.8rem; margin-top: 0.3rem;">
                            Confidence: {hazard.confidence*100:.0f}% | Keywords: {', '.join(hazard.matched_keywords[:6])}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Detailed Explanation
            if report.explanation:
                st.markdown("#### 📝 Detailed Analysis")
                st.markdown(report.explanation)

            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

            # Recommended Actions
            if report.recommended_actions:
                st.markdown("#### 🚨 Recommended Actions")
                for action in report.recommended_actions:
                    st.markdown(
                        f'<div class="action-item">{action}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

            # Additional Hazards
            if report.additional_hazards:
                st.markdown("#### ⚠️ Additional Hazards Identified")
                for h in report.additional_hazards:
                    sev_lower = h.severity.lower()
                    st.markdown(
                        f"""
                        <div class="hazard-card hazard-{sev_lower}" style="padding: 0.8rem 1.2rem;">
                            <span>{h.icon} <strong>{h.hazard_name}</strong></span>
                            <span class="severity-badge severity-{sev_lower}" style="margin-left: 0.5rem;">
                                {h.severity}
                            </span>
                            <span style="color: #94A3B8; margin-left: 0.5rem; font-size: 0.85rem;">
                                ({h.confidence*100:.0f}% confidence)
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

            # Scene Elements
            if report.scene_elements:
                st.markdown("#### 🔎 Detected Scene Elements")
                pills_html = "".join(
                    f'<span class="scene-pill">{elem}</span>'
                    for elem in report.scene_elements
                )
                st.markdown(pills_html, unsafe_allow_html=True)

            # Expandable: Raw Analysis Details
            with st.expander("🔬 Raw Analysis Details", expanded=False):
                if result.scene_analysis:
                    st.markdown("**Image Caption:**")
                    st.code(result.scene_analysis.caption)

                    st.markdown("**VQA Diagnostic Answers:**")
                    for q, a in result.scene_analysis.vqa_answers.items():
                        st.markdown(f"- **Q:** {q}")
                        st.markdown(f"  **A:** `{a}`")

        # ── Gemini Results ──────────────────────────────────────────────
        elif result.backend == "gemini" and result.gemini_result:
            gemini = result.gemini_result

            if gemini.success:
                st.markdown(
                    """
                    <div class="info-card">
                        <p style="margin: 0; font-size: 0.85rem; color: #94A3B8;">
                            ✨ <strong>Powered by Google Gemini</strong> — 
                            Native multimodal reasoning for contextual hazard analysis
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(gemini.response_text)
            else:
                st.error(f"Gemini error: {gemini.error}")

        # ── Pipeline Steps ──────────────────────────────────────────────
        st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### ⚡ Processing Pipeline")

        for step in result.steps:
            status_icon = {
                "completed": "✅",
                "running": "⏳",
                "failed": "❌",
                "pending": "⏸️",
            }.get(step.status, "❓")
            step_class = f"step-{step.status}"

            st.markdown(
                f"""
                <div class="pipeline-step {step_class}">
                    <span style="margin-right: 0.5rem;">{status_icon}</span>
                    <span style="flex: 1; font-weight: 500;">{step.name}</span>
                    <span style="color: #6C63FF; font-weight: 600;">{format_time(step.duration)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if step.detail:
                st.caption(f"  └─ {step.detail}")


# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 0.8rem; padding: 1rem 0;">
        <p>
            🛡️ <strong>Multi-Modal AI Safety Assistant</strong> — Prototype v1.0<br>
            Built with Streamlit • Hugging Face Transformers • Google Gemini API
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
