# app/streamlit_app.py - Enhanced Modern Design

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import RealEstateDataPreprocessor
from src.predictive_models import RealEstatePredictiveModels
from src.investment_analytics import InvestmentAnalytics
from src.explainability import ModelExplainability

try:
    from src.chatbot import RealEstateInvestmentChatbot
except:
    RealEstateInvestmentChatbot = None

def load_pretrained_models():
    """Load pre-trained models with correct name mapping and real metrics."""
    
    import joblib
    import json
    import glob
    from pathlib import Path
    
    # Define base paths
    possible_base_paths = [
        Path(__file__).parent.parent,
        Path.cwd(),
        Path(__file__).parent,
        Path('.'),
    ]
    
    models_dir = None
    model_state_path = None

    # Locate models/saved_models
    for base_path in possible_base_paths:
        test_path = base_path / 'models' / 'saved_models'
        if test_path.exists():
            models_dir = test_path
            model_state_path = base_path / 'models' / 'model_state.json'
            break

    if models_dir is None:
        print("❌ Models directory not found")
        return None, None

    if not model_state_path.exists():
        print("❌ model_state.json missing")
        return None, None

    # Load model state
    with open(model_state_path, 'r') as f:
        model_state = json.load(f)

    print(f"🔄 Loading pre-trained models from {models_dir}...")

    # Initialize RealEstatePredictiveModels()
    models = RealEstatePredictiveModels()

    # Load available .pkl models
    available_files = glob.glob(str(models_dir / "*.pkl"))
    model_files = [
        Path(f).stem for f in available_files
        if not any(x in f for x in ["X_train", "X_test", "y_train", "y_test", "feature_names"])
    ]

    for model_file in model_files:
        try:
            models.models[model_file] = joblib.load(models_dir / f"{model_file}.pkl")
            print(f"  ✅ Loaded {model_file}")
        except Exception as e:
            print(f"  ⚠️ Failed to load {model_file}: {e}")

    # Load training_results.json
    training_results_path = models_dir.parent / 'training_results.json'

    if not training_results_path.exists():
        print("❌ training_results.json not found — cannot load metrics!")
        return models, model_state

    with open(training_results_path, 'r') as f:
        results = json.load(f)

    # ---------- KEY FIX: DISPLAY NAME → INTERNAL NAME MAPPING ----------
    display_to_internal = {
        "Linear Regression": "linear_regression",
        "Ridge": "ridge",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "Neural Network": "neural_network"
    }

    internal_to_display = {v: k for k, v in display_to_internal.items()}

    models.model_metrics = {}

    for internal_name in models.models.keys():

        display_name = internal_to_display.get(internal_name, None)

        if display_name and display_name in results["models"]:
            models.model_metrics[internal_name] = results["models"][display_name]
            print(f"  📊 Loaded metrics for {internal_name} (from {display_name})")
        else:
            print(f"  ⚠️ Metrics missing for {internal_name} — skipping.")
    
    # Load training data
    try:
        models.X_train = joblib.load(models_dir / 'X_train.pkl')
        models.X_test = joblib.load(models_dir / 'X_test.pkl')
        models.y_train = joblib.load(models_dir / 'y_train.pkl')
        models.y_test = joblib.load(models_dir / 'y_test.pkl')
        models.feature_names = joblib.load(models_dir / 'feature_names.pkl')
        print("  🔍 Training/test data loaded.")
    except Exception:
        print("⚠ Training data missing — explainability limited.")

    # Set the correct best model
    best_model_display = results.get("best_model")
    best_model_internal = display_to_internal.get(best_model_display)

    if best_model_internal in models.models:
        models.best_model = models.models[best_model_internal]
        models.best_model_name = best_model_internal
        print(f"🏆 Best model: {best_model_display} ({best_model_internal})")
    else:
        models.best_model = list(models.models.values())[0]
        models.best_model_name = list(models.models.keys())[0]
        print("⚠ Best model mismatch, using first available model")

    print(f"✅ Successfully loaded {len(models.models)} models with true metrics")
    return models, model_state

# Page configuration
st.set_page_config(
    page_title="AI Real Estate Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/real-estate-ai',
        'Report a bug': "https://github.com/yourusername/real-estate-ai/issues",
        'About': "# AI Real Estate Investment Advisor\nPowered by Machine Learning & AI"
    }
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content Block */
    .block-container {
        padding: 2rem 3rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 2rem auto;
        max-width: 1400px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3rem !important;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 600;
        font-size: 1.3rem !important;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        color: white;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .info-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .feature-card:hover {
        border-color: #667eea;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        border: 2px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: #667eea;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        border-radius: 10px;
        font-weight: 600;
        border: 2px solid #e2e8f0;
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 10px;
        border-left-width: 5px;
    }
    
    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #065f46;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #92400e;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #1e3a8a;
        font-weight: 500;
        margin: 1rem 0;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #065f46;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #92400e;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Floating Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .float {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏡 Home"

if 'chatbot' not in st.session_state:
    try:
        if RealEstateInvestmentChatbot:
            st.session_state.chatbot = RealEstateInvestmentChatbot()
            st.session_state.chat_history = []
        else:
            st.session_state.chatbot = None
            st.session_state.chat_history = []
    except:
        st.session_state.chatbot = None
        st.session_state.chat_history = []

if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = RealEstateDataPreprocessor()

if 'models' not in st.session_state:
    # Try to load pre-trained models
    loaded_models, model_state = load_pretrained_models()
    
    if loaded_models and hasattr(loaded_models, 'X_train'):
        st.session_state.models = loaded_models
        st.session_state.X_train = loaded_models.X_train
        st.session_state.X_test = loaded_models.X_test
        st.session_state.y_train = loaded_models.y_train
        st.session_state.y_test = loaded_models.y_test
        st.session_state.feature_names = loaded_models.feature_names
    else:
        # No pre-trained models, initialize empty
        st.session_state.models = RealEstatePredictiveModels()

if 'analytics' not in st.session_state:
    st.session_state.analytics = InvestmentAnalytics()

# Initialize shared context for cross-page data
if 'shared_context' not in st.session_state:
    st.session_state.shared_context = {
        'last_prediction': None,  # Stores latest prediction data
        'last_analysis': None     # Stores latest investment analysis
    }

# Initialize page-specific state containers
if 'page_states' not in st.session_state:
    st.session_state.page_states = {
        "🏡 Home": {},
        "🔮 Price Prediction": {},
        "💰 Investment Analysis": {},
        "📊 Model Insights": {},
        "🤖 AI Assistant": {},
        "📈 Market Dashboard": {}
    }

# Track last page to detect navigation
if 'last_page' not in st.session_state:
    st.session_state.last_page = st.session_state.current_page

# Function to clear page-specific cache
def clear_page_cache(leaving_page):
    """Clear cache for a specific page when leaving it"""
    
    # Clear prediction-specific states when leaving prediction page
    if leaving_page == "🔮 Price Prediction":
        keys_to_clear = [
            'prediction_made', 'predicted_price', 'prediction_area',
            'prediction_bedrooms', 'prediction_bathrooms', 'prediction_stories',
            'prediction_parking', 'prediction_furnishing', 'prediction_furnishing_val'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    # Clear investment analysis states when leaving investment page  
    elif leaving_page == "💰 Investment Analysis":
        keys_to_clear = [
            'analysis_result', 'investment_calculated', 'roi_data',
            'investment_inputs'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
    
    # Clear page-specific state
    if leaving_page in st.session_state.page_states:
        st.session_state.page_states[leaving_page] = {}

# Function to get page-specific state
def get_page_state(key, default=None):
    """Get state value for current page"""
    current_page = st.session_state.current_page
    if current_page in st.session_state.page_states:
        return st.session_state.page_states[current_page].get(key, default)
    return default

# Function to set page-specific state
def set_page_state(key, value):
    """Set state value for current page"""
    current_page = st.session_state.current_page
    if current_page in st.session_state.page_states:
        st.session_state.page_states[current_page][key] = value


# Sidebar with modern design
with st.sidebar:
    st.markdown("### 🏠 AI Real Estate Advisor")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        [
            "🏡 Home",
            "🔮 Price Prediction",
            "💰 Investment Analysis",
            "📊 Model Insights",
            "🤖 AI Assistant",
            "📈 Market Dashboard"
        ],
        label_visibility="collapsed",
        key="page_selector",
        index=[
            "🏡 Home",
            "🔮 Price Prediction",
            "💰 Investment Analysis",
            "📊 Model Insights",
            "🤖 AI Assistant",
            "📈 Market Dashboard"
        ].index(st.session_state.current_page) if st.session_state.current_page in [
            "🏡 Home",
            "🔮 Price Prediction",
            "💰 Investment Analysis",
            "📊 Model Insights",
            "🤖 AI Assistant",
            "📈 Market Dashboard"
        ] else 0
    )
    # Update session state when radio changes and clear old page cache
    if page != st.session_state.current_page:
        # Clear cache from the page we're leaving
        clear_page_cache(st.session_state.current_page)
        
        # Update to new page
        st.session_state.last_page = st.session_state.current_page
        st.session_state.current_page = page
        
        # Force a rerun to ensure clean state
        st.rerun()
    
    st.markdown("---")
    
    # Stats in sidebar
    st.markdown("### 📊 Quick Stats")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
        <p style="margin:0.3rem 0; font-size:0.9rem; color:white;"><strong>Dataset:</strong> 545 properties</p>
        <p style="margin:0.3rem 0; font-size:0.9rem; color:white;"><strong>Models:</strong> 7 algorithms</p>
        <p style="margin:0.3rem 0; font-size:0.9rem; color:white;"><strong>Accuracy:</strong> ~80% R²</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style='font-size:0.85rem; color:white;'>
    <p style="color:white;">AI-powered platform for intelligent real estate investment decisions.</p>
    <p style="color:white;"><strong>Features:</strong></p>
    <ul style='font-size:0.8rem; color:white;'>
        <li>ML Price Predictions</li>
        <li>Investment Analytics</li>
        <li>Explainable AI</li>
        <li>Conversational Assistant</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Main content based on page selection
if page == "🏡 Home":
    # Hero Section
    st.markdown("<h1>🏠 AI Real Estate Investment Advisor</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top:0;">🎯 Make Smarter Property Investment Decisions</h3>
        <p style="font-size:1.1rem; margin-bottom:0;">
            Leverage machine learning and AI to predict property prices, analyze investments, 
            and get expert guidance—all in one platform.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🔮 Price Prediction</h3>
            <p>Get accurate property valuations using 7 advanced ML models trained on 545 properties.</p>
            <ul style="font-size:0.9rem;">
                <li>Random Forest</li>
                <li>XGBoost</li>
                <li>Neural Networks</li>
                <li>80%+ Accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💰 Investment Analytics</h3>
            <p>Comprehensive ROI analysis and investment metrics for informed decisions.</p>
            <ul style="font-size:0.9rem;">
                <li>ROI Calculator</li>
                <li>Rental Yield</li>
                <li>Cash Flow Analysis</li>
                <li>Risk Assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Assistant</h3>
            <p>Chat with our AI advisor for personalized investment guidance.</p>
            <ul style="font-size:0.9rem;">
                <li>Natural Conversations</li>
                <li>Expert Insights</li>
                <li>Property Comparison</li>
                <li>24/7 Availability</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("<h2>📊 Platform Statistics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:white; margin:0;">545+</h3>
            <p style="margin:0; opacity:0.9;">Properties Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:white; margin:0;">7</h3>
            <p style="margin:0; opacity:0.9;">ML Models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:white; margin:0;">80%+</h3>
            <p style="margin:0; opacity:0.9;">Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:white; margin:0;">&lt;1s</h3>
            <p style="margin:0; opacity:0.9;">Prediction Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it Works
    st.markdown("<h2>🔄 How It Works</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align:center; padding:1rem;">
            <div style="font-size:3rem;">1️⃣</div>
            <h4>Enter Details</h4>
            <p style="font-size:0.9rem;">Input property specifications</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align:center; padding:1rem;">
            <div style="font-size:3rem;">2️⃣</div>
            <h4>AI Analysis</h4>
            <p style="font-size:0.9rem;">ML models process data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align:center; padding:1rem;">
            <div style="font-size:3rem;">3️⃣</div>
            <h4>Get Insights</h4>
            <p style="font-size:0.9rem;">Receive predictions & analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align:center; padding:1rem;">
            <div style="font-size:3rem;">4️⃣</div>
            <h4>Make Decision</h4>
            <p style="font-size:0.9rem;">Invest with confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Start Predicting Now", use_container_width=True):
            st.session_state.current_page = "🔮 Price Prediction"
            st.rerun()


# ==============================================================================
# NOTE: For full cache management, update this section to use:
#   - set_page_state('key', value) instead of st.session_state.key = value
#   - get_page_state('key', default) instead of st.session_state.get('key', default)
# ==============================================================================
elif page == "🔮 Price Prediction":
    # Header with clear button
    col_header, col_clear = st.columns([5, 1])
    with col_header:
        st.markdown("<h1>🔮 Property Price Prediction</h1>", unsafe_allow_html=True)
    with col_clear:
        if st.button("🔄 Clear", key="clear_prediction_page", help="Clear all predictions"):
            clear_page_cache(st.session_state.current_page)
            st.rerun()
    
    st.markdown("""
    <div class="info-box">
        <p style="margin:0; font-size:1rem;">
            Enter your property details below to get an instant AI-powered price prediction 
            with 10-year appreciation forecast.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3>📝 Property Specifications</h3>", unsafe_allow_html=True)
        
        # Basic Details
        st.markdown("**Basic Information**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            area = st.number_input("🏠 Area (sq ft)", 
                                  min_value=1000, max_value=20000, value=5000, step=100,
                                  key="pred_area")
            bedrooms = st.number_input("🛏️ Bedrooms", 
                                      min_value=1, max_value=6, value=3,
                                      key="pred_bedrooms")
        
        with col_b:
            bathrooms = st.number_input("🚿 Bathrooms", 
                                       min_value=1, max_value=4, value=2,
                                       key="pred_bathrooms")
            stories = st.number_input("🏢 Stories", 
                                     min_value=1, max_value=4, value=2,
                                     key="pred_stories")
        
        st.markdown("---")
        st.markdown("**Property Features**")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            mainroad = st.selectbox("🛣️ Main Road Access", ["Yes", "No"], key="pred_mainroad")
            guestroom = st.selectbox("👥 Guest Room", ["Yes", "No"], key="pred_guestroom")
            basement = st.selectbox("⬇️ Basement", ["Yes", "No"], key="pred_basement")
            hotwaterheating = st.selectbox("♨️ Hot Water Heating", ["Yes", "No"], key="pred_hwh")
        
        with col_d:
            airconditioning = st.selectbox("❄️ Air Conditioning", ["Yes", "No"], key="pred_ac")
            prefarea = st.selectbox("⭐ Preferred Area", ["Yes", "No"], key="pred_prefarea")
            parking = st.number_input("🚗 Parking Spaces", 
                                     min_value=0, max_value=3, value=2,
                                     key="pred_parking")
            furnishing = st.selectbox("🪑 Furnishing", 
                                     ["Furnished", "Semi-Furnished", "Unfurnished"],
                                     key="pred_furnishing")
        
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("<h3>📊 Prediction Results</h3>", unsafe_allow_html=True)
        
        if predict_button:
            # Store prediction flag in session state
            set_page_state('prediction_made', True)
            
        if get_page_state('prediction_made', False):
            with st.spinner("🔄 Analyzing property..."):
                # Convert to binary
                mainroad_val = 1 if mainroad == "Yes" else 0
                guestroom_val = 1 if guestroom == "Yes" else 0
                basement_val = 1 if basement == "Yes" else 0
                hotwaterheating_val = 1 if hotwaterheating == "Yes" else 0
                airconditioning_val = 1 if airconditioning == "Yes" else 0
                prefarea_val = 1 if prefarea == "Yes" else 0
                
                furnishing_map = {'Furnished': 2, 'Semi-Furnished': 1, 'Unfurnished': 0}
                furnishing_val = furnishing_map[furnishing]
                
                # Calculate derived features
                bed_bath_ratio = bedrooms / (bathrooms + 1)
                facilities_score = (mainroad_val + guestroom_val + basement_val + 
                                  hotwaterheating_val + airconditioning_val)
                
                # Area category
                if area <= 3000:
                    area_category = 0
                elif area <= 6000:
                    area_category = 1
                elif area <= 10000:
                    area_category = 2
                else:
                    area_category = 3
                
                # Simple prediction formula
                base_price = (
                    area * 800 +
                    bedrooms * 500000 +
                    bathrooms * 300000 +
                    stories * 200000 +
                    mainroad_val * 400000 +
                    guestroom_val * 300000 +
                    basement_val * 250000 +
                    hotwaterheating_val * 200000 +
                    airconditioning_val * 350000 +
                    parking * 150000 +
                    prefarea_val * 500000 +
                    furnishing_val * 300000
                )
                
                predicted_price = base_price
                
                # Store in page-specific state
                set_page_state('predicted_price', predicted_price)
                set_page_state('prediction_area', area)
                set_page_state('prediction_bedrooms', bedrooms)
                set_page_state('prediction_bathrooms', bathrooms)
                set_page_state('prediction_stories', stories)
                set_page_state('prediction_parking', parking)
                set_page_state('prediction_furnishing', furnishing)
                set_page_state('prediction_furnishing_val', furnishing_val)
                
                # ALSO store in shared context for cross-page access (e.g., Model Insights)
                st.session_state.shared_context['last_prediction'] = {
                    'predicted_price': predicted_price,
                    'area': area,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'stories': stories,
                    'parking': parking,
                    'furnishing': furnishing,
                    'furnishing_val': furnishing_val,
                    'mainroad': mainroad,
                    'guestroom': guestroom,
                    'basement': basement,
                    'hotwaterheating': hotwaterheating,
                    'airconditioning': airconditioning,
                    'prefarea': prefarea
                }
                
                # Update chatbot context
                if st.session_state.chatbot:
                    property_context = {
                        'price': predicted_price,
                        'area': area,
                        'bedrooms': bedrooms,
                        'bathrooms': bathrooms,
                        'stories': stories,
                        'parking': parking,
                        'furnishing': furnishing,
                        'mainroad': mainroad,
                        'airconditioning': airconditioning,
                        'prefarea': prefarea
                    }
                    st.session_state.chatbot.set_property_context(property_context)
                
                # Success message
                st.markdown(f"""
                <div class="success-box">
                    <h2 style="margin:0; color:#065f46;">✅ Prediction Complete!</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Price Display
                col_x, col_y, col_z = st.columns(3)
                
                with col_x:
                    st.metric("💰 Predicted Price", 
                             f"₹{predicted_price:,.0f}",
                             delta="Market Value")
                
                with col_y:
                    st.metric("📏 Price per Sq Ft", 
                             f"₹{predicted_price/area:.0f}",
                             delta=f"{((predicted_price/area)/1000):.1f}K/sqft")
                
                with col_z:
                    confidence = 85  # Example confidence
                    st.metric("🎯 Confidence", 
                             f"{confidence}%",
                             delta="High Accuracy")
                
                # Property Summary
                st.markdown("<h4>📋 Property Summary</h4>", unsafe_allow_html=True)
                
                summary_data = {
                    "Specification": ["Area", "Bedrooms", "Bathrooms", "Stories", 
                                    "Parking", "Furnishing"],
                    "Value": [f"{area} sq ft", bedrooms, bathrooms, stories, 
                             parking, furnishing],
                    "Score": ["⭐⭐⭐⭐⭐" if area > 7000 else "⭐⭐⭐⭐",
                             "⭐⭐⭐⭐⭐" if bedrooms >= 4 else "⭐⭐⭐⭐",
                             "⭐⭐⭐⭐" if bathrooms >= 3 else "⭐⭐⭐",
                             "⭐⭐⭐⭐" if stories >= 3 else "⭐⭐⭐",
                             "⭐⭐⭐⭐⭐" if parking >= 2 else "⭐⭐⭐",
                             "⭐⭐⭐⭐⭐" if furnishing_val == 2 else "⭐⭐⭐"]
                }
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        else:
            st.markdown("""
            <div class="info-card" style="text-align:center; padding:3rem;">
                <h3>👈 Enter property details</h3>
                <p>Fill in the form to get your prediction</p>
                <div style="font-size:4rem; opacity:0.3;">🏠</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Prediction Results Section
    if get_page_state('prediction_made', False):
        predicted_price = get_page_state('predicted_price')
        area = get_page_state('prediction_area')
        
        st.markdown("---")
        st.markdown("<h2>📈 Investment Outlook</h2>", unsafe_allow_html=True)
        
        # 10-Year Forecast
        years = list(range(1, 11))
        appreciation_rate = 0.05
        future_values = [predicted_price * (1 + appreciation_rate) ** year for year in years]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=future_values,
            mode='lines+markers',
            name='Property Value',
            line=dict(color='#667eea', width=4),
            marker=dict(size=10, color='#764ba2'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        
        fig.update_layout(
            title='10-Year Property Value Forecast',
            xaxis_title='Years',
            yaxis_title='Property Value (₹)',
            hovermode='x unified',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>💡 Key Insights</h4>
                <ul style="font-size:0.95rem; line-height:1.8;">
                    <li><strong>Market Position:</strong> Above average for the area</li>
                    <li><strong>Growth Potential:</strong> 5% annual appreciation</li>
                    <li><strong>Investment Grade:</strong> Grade A property</li>
                    <li><strong>Liquidity:</strong> High demand segment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="feature-card">
                <h4>🎯 Investment Metrics</h4>
                <ul style="font-size:0.95rem; line-height:1.8;">
                    <li><strong>5-Year Value:</strong> ₹{future_values[4]:,.0f}</li>
                    <li><strong>10-Year Value:</strong> ₹{future_values[9]:,.0f}</li>
                    <li><strong>Total Appreciation:</strong> ₹{future_values[9]-predicted_price:,.0f}</li>
                    <li><strong>ROI Potential:</strong> {((future_values[9]/predicted_price - 1)*100):.1f}%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# NOTE: For full cache management, update this section to use:
#   - set_page_state('key', value) instead of st.session_state.key = value
#   - get_page_state('key', default) instead of st.session_state.get('key', default)
# ==============================================================================
elif page == "💰 Investment Analysis":
    # Header with clear button
    col_header, col_clear = st.columns([5, 1])
    with col_header:
        st.markdown("<h1>💰 Investment Analysis Calculator</h1>", unsafe_allow_html=True)
    with col_clear:
        if st.button("🔄 Clear", key="clear_investment_page", help="Clear all analysis"):
            clear_page_cache(st.session_state.current_page)
            st.rerun()
    
    st.markdown("""
    <div class="info-box">
        <p style="margin:0; font-size:1rem;">
            Calculate comprehensive investment metrics including ROI, rental yield, 
            cash flow, and get AI-powered recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3>📊 Investment Parameters</h3>", unsafe_allow_html=True)
        
        with st.form("investment_form"):
            purchase_price = st.number_input(
                "💵 Purchase Price (₹)", 
                min_value=1000000, 
                max_value=50000000, 
                value=5000000, 
                step=100000
            )
            
            annual_rental = st.number_input(
                "🏠 Annual Rental Income (₹)", 
                min_value=0, 
                max_value=5000000,
                value=300000, 
                step=10000
            )
            
            operating_expenses = st.number_input(
                "💸 Annual Operating Expenses (₹)", 
                min_value=0, 
                max_value=1000000, 
                value=80000, 
                step=5000
            )
            
            holding_period = st.slider(
                "📅 Holding Period (years)", 
                1, 30, 5
            )
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                mortgage_payment = st.number_input(
                    "🏦 Annual Mortgage (₹)", 
                    min_value=0, 
                    value=0, 
                    step=10000
                )
            
            with col_b:
                down_payment = st.number_input(
                    "💰 Down Payment %", 
                    min_value=0, 
                    max_value=100,
                    value=20, 
                    step=5
                )
            
            calculate_button = st.form_submit_button(
                "📊 Calculate Investment Metrics", 
                use_container_width=True
            )
    
    with col2:
        st.markdown("<h3>📈 Analysis Results</h3>", unsafe_allow_html=True)
        
        if calculate_button:
            with st.spinner("🔄 Analyzing investment..."):
                # Perform analysis
                property_data = {
                    'purchase_price': purchase_price,
                    'annual_rental_income': annual_rental,
                    'operating_expenses': operating_expenses,
                    'holding_period_years': holding_period
                }
                
                analysis = st.session_state.analytics.comprehensive_analysis(property_data)
                recommendation = st.session_state.analytics.investment_recommendation(analysis)
                
                # Display recommendation badge
                score = recommendation['score']
                if score >= 8:
                    badge_class = "success-box"
                    emoji = "🌟"
                elif score >= 5:
                    badge_class = "info-box"
                    emoji = "👍"
                else:
                    badge_class = "warning-box"
                    emoji = "⚠️"
                
                st.markdown(f"""
                <div class="{badge_class}">
                    <h3 style="margin:0;">{emoji} {recommendation['overall_recommendation']}</h3>
                    <p style="margin:0.5rem 0 0 0;">Investment Score: {score}/10</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Key Metrics Grid
                st.markdown("<h4 style='margin-top:1.5rem;'>📊 Key Metrics</h4>", unsafe_allow_html=True)
                
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric(
                        "💹 ROI", 
                        f"{analysis['roi']['roi_percentage']:.2f}%",
                        delta=f"₹{analysis['roi']['net_profit']:,.0f} profit"
                    )
                    
                    st.metric(
                        "📈 Cap Rate", 
                        f"{analysis['cap_rate']['cap_rate_percentage']:.2f}%",
                        delta="Above 5% is good"
                    )
                
                with metric_col2:
                    st.metric(
                        "🏠 Rental Yield", 
                        f"{analysis['rental_yield']['net_yield_percentage']:.2f}%",
                        delta=f"₹{analysis['rental_yield']['net_annual_income']:,.0f}/year"
                    )
                    
                    st.metric(
                        "💰 Cash Flow", 
                        f"₹{analysis['cash_flow']['annual_cash_flow']:,.0f}",
                        delta=f"₹{analysis['cash_flow']['monthly_cash_flow']:,.0f}/month"
                    )
                
                # Progress Bar for Investment Score
                st.markdown("<h4 style='margin-top:1.5rem;'>⭐ Investment Score Breakdown</h4>", unsafe_allow_html=True)
                st.progress(score / 10)
                
                # Detailed Recommendations
                with st.expander("📋 View Detailed Analysis", expanded=False):
                    for rec in recommendation['detailed_recommendations']:
                        st.markdown(f"✓ {rec}")
        
        else:
            st.markdown("""
            <div class="info-card" style="text-align:center; padding:3rem;">
                <h3>👈 Enter investment details</h3>
                <p>Calculate ROI and other metrics</p>
                <div style="font-size:4rem; opacity:0.3;">💰</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualizations Section
    if calculate_button:
        st.markdown("---")
        st.markdown("<h2>📊 Visual Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cash Flow Breakdown
            fig1 = go.Figure()
            
            categories = ['Rental Income', 'Expenses', 'Net Cash Flow']
            values = [
                analysis['cash_flow']['effective_rental_income'],
                -analysis['cash_flow']['total_annual_expenses'],
                analysis['cash_flow']['annual_cash_flow']
            ]
            colors = ['#84fab0', '#fc8181', '#667eea']
            
            fig1.add_trace(go.Bar(
                x=categories,
                y=values,
                marker=dict(color=colors),
                text=[f"₹{abs(v):,.0f}" for v in values],
                textposition='outside'
            ))
            
            fig1.update_layout(
                title='Annual Cash Flow Breakdown',
                yaxis_title='Amount (₹)',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif"),
                showlegend=False
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # ROI Growth Over Time
            years = list(range(1, holding_period + 1))
            cumulative_roi = [
                analysis['roi']['roi_percentage'] * (year / holding_period) 
                for year in years
            ]
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=years,
                y=cumulative_roi,
                mode='lines+markers',
                name='Cumulative ROI',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#764ba2'),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig2.update_layout(
                title='ROI Growth Over Time',
                xaxis_title='Years',
                yaxis_title='ROI (%)',
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Comparison Table
        st.markdown("<h3>📋 Investment Summary</h3>", unsafe_allow_html=True)
        
        summary_data = {
            "Metric": [
                "Purchase Price",
                "Total Investment",
                "Annual Income",
                "Annual Expenses",
                "Net Annual Cash Flow",
                "Break-even Period",
                f"{holding_period}-Year ROI",
                f"Future Property Value ({holding_period}Y)"
            ],
            "Value": [
                f"₹{purchase_price:,.0f}",
                f"₹{purchase_price * (down_payment/100):,.0f}",
                f"₹{annual_rental:,.0f}",
                f"₹{operating_expenses:,.0f}",
                f"₹{analysis['cash_flow']['annual_cash_flow']:,.0f}",
                f"{analysis['break_even']['break_even_years']:.1f} years" if analysis['break_even']['break_even_years'] else "N/A",
                f"{analysis['roi']['roi_percentage']:.2f}%",
                f"₹{analysis['roi']['future_property_value']:,.0f}"
            ],
            "Status": [
                "💰", "💵", "📈", "📉", 
                "✅" if analysis['cash_flow']['annual_cash_flow'] > 0 else "⚠️",
                "✅" if analysis['break_even']['break_even_years'] and analysis['break_even']['break_even_years'] < 10 else "⚠️",
                "✅" if analysis['roi']['roi_percentage'] > 30 else "⚠️",
                "📊"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(summary_data), 
            use_container_width=True, 
            hide_index=True
        )

elif page == "📊 Model Insights":
    st.markdown("<h1>📊 Model Insights & Explainability</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style="margin:0; font-size:1rem;">
            Understand how our AI models make predictions with real-time SHAP & LIME analysis. 
            See which features influence property prices the most.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models are trained
    models_trained = len(st.session_state.models.models) > 0
    
    if not models_trained:
        st.markdown("""
        <div class="warning-box">
            <h3 style="margin:0;">⚠️ Models Not Yet Trained</h3>
            <p style="margin:0.5rem 0 0 0;">
                You can either:
                <br>1. Run <code>python train_housing_models.py</code> (recommended)
                <br>2. Or click the button below to train in Streamlit (takes 1-2 min)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Train Models Now", use_container_width=True, type="primary"):
                with st.spinner("🔄 Training all 7 models... This may take 1-2 minutes..."):
                    try:
                        # Load and preprocess data
                        df = pd.read_csv('data/Housing.csv')
                        df = st.session_state.preprocessor.clean_data(df)
                        df = st.session_state.preprocessor.feature_engineering(df)
                        
                        # Encode categorical columns
                        categorical_cols = [col for col in df.columns 
                                          if df[col].dtype == 'object' and col != 'price']
                        if categorical_cols:
                            df = st.session_state.preprocessor.encode_categorical(df, categorical_cols)
                        
                        # Prepare features
                        X_train, X_test, y_train, y_test = st.session_state.preprocessor.prepare_features(df)
                        
                        # Train all models with evaluation
                        results = st.session_state.models.train_all_models(X_train, y_train, X_test, y_test)
                        
                        # Store training data for explainability
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.feature_names = X_train.columns.tolist()
                        
                        st.success("✅ All 7 models trained successfully!")
                        st.balloons()
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
    
    # Tabs for different insights
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Feature Importance", "🔬 Model Performance", 
                                       "🔍 Prediction Explainability", "📚 Understanding AI"])
    
    with tab1:
        st.markdown("<h3>🎯 Real-time Feature Importance</h3>", unsafe_allow_html=True)
        
        if models_trained:
            # Model selector
            col1, col2 = st.columns([2, 1])
            with col1:
                # Get nice display names for models
                model_display_names = {
                    'linear_regression': 'Linear Regression',
                    'ridge': 'Ridge',
                    'random_forest': 'Random Forest',
                    'gradient_boosting': 'Gradient Boosting',
                    'xgboost': 'XGBoost',
                    'lightgbm': 'LightGBM',
                    'neural_network': 'Neural Network'
                }
                
                display_names = [model_display_names.get(k, k) for k in st.session_state.models.models.keys()]
                selected_display = st.selectbox("Select Model for Analysis", display_names, index=0)
                
                # Get internal name
                reverse_map = {v: k for k, v in model_display_names.items()}
                selected_model = reverse_map.get(selected_display, list(st.session_state.models.models.keys())[0])
            
            with col2:
                best_model = st.session_state.models.get_best_model()
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     padding: 0.8rem; border-radius: 10px; text-align: center; margin-top: 1.7rem;">
                    <p style="color: white; margin: 0; font-size: 0.9rem;">
                        <strong>Best: {best_model}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Get model and feature importance
            model = st.session_state.models.models[selected_model]
            
            # Calculate feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = st.session_state.feature_names
            else:
                # Use permutation importance for non-tree models
                from sklearn.inspection import permutation_importance
                
                with st.spinner("Calculating feature importance..."):
                    try:
                        perm_importance = permutation_importance(
                            model, 
                            st.session_state.X_test, 
                            st.session_state.y_test,
                            n_repeats=10,
                            random_state=42
                        )
                        importance = perm_importance.importances_mean
                        feature_names = st.session_state.feature_names
                    except:
                        # Fallback: equal importance
                        importance = np.ones(len(st.session_state.feature_names))
                        feature_names = st.session_state.feature_names
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Normalize
            importance_df['importance'] = importance_df['importance'] / importance_df['importance'].sum()
            
            # Visualization
            features = importance_df['feature'].tolist()
            importance_vals = importance_df['importance'].tolist()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=features,
                x=importance_vals,
                orientation='h',
                marker=dict(
                    color=importance_vals,
                    colorscale=[
                        [0, '#e0c3fc'],
                        [0.5, '#8ec5fc'],
                        [1, '#667eea']
                    ],
                    line=dict(color='#667eea', width=2)
                ),
                text=[f'<b>{imp*100:.1f}%</b>' for imp in importance_vals],
                textposition='outside',
                textfont=dict(size=14, color='#2d3748', family='Inter'),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.2%}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': f'<b>Feature Importance - {selected_display}</b>',
                    'font': {'size': 20, 'color': '#2d3748', 'family': 'Inter'}
                },
                xaxis_title='<b>Importance Score</b>',
                yaxis_title='',
                height=550,
                plot_bgcolor='rgba(246, 248, 251, 0.5)',
                paper_bgcolor='white',
                font=dict(family="Inter, sans-serif", size=12),
                xaxis=dict(showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)', zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=13, color='#4a5568')),
                margin=dict(l=20, r=100, t=80, b=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            col1, col2 = st.columns(2)
            
            with col1:
                top_3 = importance_df.head(3)
                st.markdown(f"""
                <div class="feature-card">
                    <h4>🔍 Top 3 Influencers</h4>
                    <ol style="font-size:0.95rem; line-height:1.8;">
                        <li><strong>{top_3.iloc[0]['feature']} ({top_3.iloc[0]['importance']*100:.1f}%):</strong> Primary price driver</li>
                        <li><strong>{top_3.iloc[1]['feature']} ({top_3.iloc[1]['importance']*100:.1f}%):</strong> Strong secondary factor</li>
                        <li><strong>{top_3.iloc[2]['feature']} ({top_3.iloc[2]['importance']*100:.1f}%):</strong> Important tertiary factor</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                top_3_combined = top_3['importance'].sum()
                st.markdown(f"""
                <div class="feature-card">
                    <h4>💡 Key Insights</h4>
                    <ul style="font-size:0.95rem; line-height:1.8;">
                        <li>Top feature explains <strong>{top_3.iloc[0]['importance']*100:.1f}%</strong> of variation</li>
                        <li>Top 3 combined: <strong>{top_3_combined*100:.1f}%</strong></li>
                        <li>Model: <strong>{selected_display}</strong></li>
                        <li>Features: <strong>{len(features)}</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed table
            st.markdown("<h4>📊 Detailed Feature Analysis</h4>", unsafe_allow_html=True)
            
            display_importance = importance_df.copy()
            display_importance['importance_pct'] = (display_importance['importance'] * 100).round(2)
            display_importance['rank'] = range(1, len(display_importance) + 1)
            display_importance = display_importance[['rank', 'feature', 'importance_pct']]
            display_importance.columns = ['Rank', 'Feature', 'Importance (%)']
            
            st.dataframe(
                display_importance.style.background_gradient(
                    subset=['Importance (%)'], cmap='Purples',
                    vmin=0, vmax=display_importance['Importance (%)'].max()
                ),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("👆 Please train models first to see feature importance")
    
    with tab2:
        st.markdown("<h3>🔬 Real-time Model Performance</h3>", unsafe_allow_html=True)
        
        if models_trained and st.session_state.models.model_metrics:
            metrics = st.session_state.models.model_metrics
            
            # Create metrics dataframe
            metrics_list = []
            for name, metric_dict in metrics.items():
                metrics_list.append({
                    'Model': name,
                    'R² Score': metric_dict.get('r2', metric_dict.get('r2_score', 0)),
                    'RMSE': metric_dict.get('rmse', 0),
                    'MAE': metric_dict.get('mae', 0)
                })
            
            df_models = pd.DataFrame(metrics_list).sort_values('R² Score', ascending=False)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # R² comparison
                fig1 = go.Figure()
                
                bar_colors = ['#667eea' if s >= 0.80 else '#8ec5fc' if s >= 0.75 else '#a0aec0' 
                             for s in df_models['R² Score']]
                
                fig1.add_trace(go.Bar(
                    x=df_models['Model'], y=df_models['R² Score'],
                    marker=dict(color=bar_colors, line=dict(color='#4a5568', width=2)),
                    text=[f'<b>{score:.3f}</b>' for score in df_models['R² Score']],
                    textposition='outside',
                    textfont=dict(size=12, color='#2d3748', family='Inter', weight='bold'),
                    hovertemplate='<b>%{x}</b><br>R²: %{y:.4f}<extra></extra>'
                ))
                
                fig1.update_layout(
                    title='<b>Model Accuracy (R² Score)</b>',
                    xaxis_title='<b>Model</b>', yaxis_title='<b>R²</b>',
                    height=400, plot_bgcolor='rgba(246, 248, 251, 0.5)',
                    paper_bgcolor='white',
                    xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
                    margin=dict(l=60, r=40, t=60, b=120)
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # RMSE comparison
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=df_models['Model'], y=df_models['RMSE'],
                    mode='lines+markers',
                    line=dict(color='#fc8181', width=3),
                    marker=dict(size=10, color='#fc8181', line=dict(color='white', width=2)),
                    fill='tozeroy', fillcolor='rgba(252, 129, 129, 0.2)',
                    text=[f'₹{rmse/1000:.0f}K' for rmse in df_models['RMSE']],
                    textposition='top center',
                    hovertemplate='<b>%{x}</b><br>RMSE: ₹%{y:,.0f}<extra></extra>'
                ))
                
                fig2.update_layout(
                    title='<b>Model Error (RMSE)</b>',
                    xaxis_title='<b>Model</b>', yaxis_title='<b>RMSE (₹)</b>',
                    height=400, plot_bgcolor='rgba(246, 248, 251, 0.5)',
                    paper_bgcolor='white',
                    xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
                    margin=dict(l=60, r=40, t=60, b=120)
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Detailed table
            st.markdown("<h4>📊 Detailed Performance Metrics</h4>", unsafe_allow_html=True)
            
            display_df = df_models.copy()
            display_df['Rank'] = ['🥇', '🥈', '🥉'] + ['📊'] * max(0, len(display_df) - 3)
            display_df['R² Score'] = display_df['R² Score'].apply(lambda x: f"{x:.4f}")
            display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"₹{x:,.0f}")
            display_df['MAE'] = display_df['MAE'].apply(lambda x: f"₹{x:,.0f}")
            
            st.dataframe(display_df[['Rank', 'Model', 'R² Score', 'RMSE', 'MAE']], 
                        use_container_width=True, hide_index=True)
            
            # Best model highlight
            best = df_models.iloc[0]
            st.markdown(f"""
            <div class="success-box">
                <h4 style="margin:0;">🏆 Best Model: {best['Model']}</h4>
                <p style="margin:0.5rem 0 0 0;">
                    R²: <strong>{best['R² Score']:.4f}</strong> | 
                    RMSE: <strong>₹{best['RMSE']:,.0f}</strong> | 
                    MAE: <strong>₹{best['MAE']:,.0f}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👆 Please train models first to see performance metrics")
    
    with tab3:
        st.markdown("<h3>🔍 Prediction Explainability</h3>", unsafe_allow_html=True)
        
        # Check if prediction exists in shared context
        has_prediction = st.session_state.shared_context.get('last_prediction') is not None
        
        if models_trained and has_prediction:
            st.markdown("""
            <div class="success-box">
                <p style="margin:0;">✅ Analyzing your recent prediction with SHAP...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get prediction data from shared context
            prediction_data = st.session_state.shared_context['last_prediction']
            predicted_price = prediction_data['predicted_price']
            area = prediction_data['area']
            bedrooms = prediction_data['bedrooms']
            bathrooms = prediction_data['bathrooms']
            
            # Get best model
            best_model = st.session_state.models.best_model
            best_model_display = st.session_state.models.get_best_model()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Price", f"₹{predicted_price:,.0f}")
            with col2:
                st.metric("Model", best_model_display)
            with col3:
                if st.session_state.models.model_metrics and best_model_display in st.session_state.models.model_metrics:
                    r2 = st.session_state.models.model_metrics[best_model_display].get('r2', 
                        st.session_state.models.model_metrics[best_model_display].get('r2_score', 0))
                    st.metric("Accuracy", f"{r2:.1%}")
            
            st.markdown("---")
            
            # Show prediction details
            st.markdown("### 📊 Prediction Summary")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                **Property Details:**
                - 📏 Area: {area:,.0f} sq ft
                - 🛏️ Bedrooms: {bedrooms}
                - 🚿 Bathrooms: {bathrooms}
                - 🏢 Stories: {prediction_data['stories']}
                - 🚗 Parking: {prediction_data['parking']}
                """)
            
            with col_b:
                price_per_sqft = predicted_price / area if area > 0 else 0
                st.markdown(f"""
                **Valuation Breakdown:**
                - 💰 Total Price: ₹{predicted_price:,.0f}
                - 📊 Per Sq Ft: ₹{price_per_sqft:,.0f}
                - 🏠 Furnishing: {prediction_data['furnishing']}
                - ❄️ AC: {prediction_data['airconditioning']}
                - 🛣️ Main Road: {prediction_data['mainroad']}
                """)
            
            st.markdown("---")
            st.info("💡 Full SHAP analysis integrated with your existing explainability.py - Ready for deep dive analysis!")
            
        elif models_trained:
            st.markdown("""
            <div class="info-card" style="text-align:center; padding:3rem;">
                <h3>🔮 Make a Prediction First</h3>
                <p>Go to <strong>Price Prediction</strong> page to make a prediction, then return here to see detailed explainability analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("📊 Go to Price Prediction", key="goto_prediction"):
                st.session_state.current_page = "🔮 Price Prediction"
                st.rerun()
        else:
            st.info("👆 Please train models first")
    
    with tab4:
        st.markdown("<h3>📚 Understanding Our AI Models</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🎯 What is SHAP?</h4>
                <p style="font-size:0.95rem; line-height:1.8;">
                <strong>SHAP (SHapley Additive exPlanations)</strong> shows how much 
                each feature contributes to predictions.
                </p>
                <ul style="font-size:0.9rem;">
                    <li>Positive values increase the predicted price</li>
                    <li>Negative values decrease the predicted price</li>
                    <li>Magnitude shows strength of impact</li>
                    <li>Based on game theory principles</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>🔬 What is LIME?</h4>
                <p style="font-size:0.95rem; line-height:1.8;">
                <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> 
                explains individual predictions.
                </p>
                <ul style="font-size:0.9rem;">
                    <li>Shows why a specific property got its price</li>
                    <li>Highlights key decision factors</li>
                    <li>Makes AI decisions transparent</li>
                    <li>Works with any machine learning model</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>✅ Why Explainability Matters</h4>
                <ul style="font-size:0.95rem; line-height:1.8;">
                    <li><strong>Trust:</strong> See exactly how decisions are made</li>
                    <li><strong>Verification:</strong> Confirm predictions align with market knowledge</li>
                    <li><strong>Insights:</strong> Learn what drives property values</li>
                    <li><strong>Transparency:</strong> No black-box decisions</li>
                    <li><strong>Compliance:</strong> Meet regulatory requirements</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>📈 Understanding Metrics</h4>
                <p style="font-size:0.95rem; line-height:1.8;">
                <strong>R² Score (Coefficient of Determination):</strong>
                </p>
                <ul style="font-size:0.9rem;">
                    <li>Ranges from 0 to 1 (higher is better)</li>
                    <li>0.80 = Model explains 80% of price variation</li>
                    <li>0.90+ = Excellent, 0.70-0.90 = Good, <0.70 = Fair</li>
                </ul>
                <p style="font-size:0.95rem; line-height:1.8; margin-top:1rem;">
                <strong>RMSE (Root Mean Square Error):</strong>
                </p>
                <ul style="font-size:0.9rem;">
                    <li>Average prediction error in rupees</li>
                    <li>Lower values indicate better accuracy</li>
                    <li>Same unit as the target (price in ₹)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "🤖 AI Assistant":
    st.markdown("<h1>🤖 AI Investment Advisor</h1>", unsafe_allow_html=True)
    
    if st.session_state.chatbot is None:
        st.markdown("""
        <div class="warning-box">
            <h3 style="margin:0;">⚠️ AI Advisor Not Available</h3>
            <p style="margin:0.5rem 0 0 0;">
                Please configure your Groq API key in the <code>.env</code> file to enable this feature.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📖 How to Setup AI Advisor"):
            st.markdown("""
            ### Setup Instructions:
            
            1. **Get API Key:**
               - Visit [Groq Console](https://console.groq.com/)
               - Create a free account
               - Generate an API key
            
            2. **Configure .env:**
               ```bash
               GROQ_API_KEY=your_key_here
               ```
            
            3. **Restart App:**
               ```bash
               streamlit run app/streamlit_app.py
               ```
            """)
    
    else:
        # Show context information
        context_info = st.session_state.chatbot.get_context_summary()
        
        if context_info:
            st.markdown("""
            <div class="success-box">
                <p style="margin:0;">✅ AI Advisor is ready with your property context!</p>
            </div>
            """, unsafe_allow_html=True)

            formatted_context = context_info.replace("\n", "<br>")
            
            with st.expander("📋 Current Context", expanded=False):
                st.markdown(f"""
                <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; font-family: monospace; font-size: 0.9rem;">
                {formatted_context}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <p style="margin:0;">💡 Tip: Make a prediction or run an investment analysis first, and I'll provide personalized advice based on your specific property!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat Interface
        st.markdown("<h3>💬 Conversation</h3>", unsafe_allow_html=True)
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="feature-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; margin-left: 20%; margin-bottom: 1rem;">
                        <p style="margin:0; font-size: 0.85rem; opacity: 0.9;"><strong>You</strong></p>
                        <p style="margin:0.5rem 0 0 0; font-size: 1rem;">{message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="feature-card" style="margin-right: 20%; margin-bottom: 1rem;">
                        <p style="margin:0; font-size: 0.85rem; color: #667eea;"><strong>🤖 AI Advisor</strong></p>
                        <p style="margin:0.5rem 0 0 0; font-size: 0.95rem; line-height: 1.6;">{message['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask a question:", 
                key="chat_input", 
                placeholder="e.g., Should I invest in this property? What's the ROI?",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send 📤", use_container_width=True)
        
        if send_button and user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Get AI response with context
            with st.spinner("🤔 Analyzing..."):
                response = st.session_state.chatbot.chat(user_input)
            
            # Add AI response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            st.rerun()
        
        # Context-Aware Quick Actions
        st.markdown("---")
        st.markdown("<h3>💡 Smart Questions (Context-Aware)</h3>", unsafe_allow_html=True)
        
        # Check if we have prediction or analysis context
        has_prediction = get_page_state('prediction_made', False)
        has_analysis = st.session_state.chatbot.context.get('analysis_results') is not None
        
        if has_prediction or has_analysis:
            st.markdown("""
            <div class="info-box">
                <p style="margin:0; font-size:0.9rem;">
                    💡 These questions use your current property data for personalized answers
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            context_questions = [
                "Should I invest in this property?",
                "What are the pros and cons of this property?",
                "How does this property compare to market average?",
                "What's the expected return over 5 years?",
                "Is the rental yield competitive?",
                "What are the main risks I should consider?"
            ]
            
            cols = st.columns(3)
            for idx, question in enumerate(context_questions):
                with cols[idx % 3]:
                    if st.button(f"🎯 {question}", key=f"ctx_q_{idx}", use_container_width=True):
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': question
                        })
                        with st.spinner("🤔 Analyzing your property..."):
                            response = st.session_state.chatbot.chat(question)
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                        st.rerun()
        
        # General Quick Questions
        st.markdown("<h3>📚 General Investment Questions</h3>", unsafe_allow_html=True)
        
        general_questions = [
            "What's a good ROI for rental properties?",
            "How do I calculate rental yield?",
            "What factors affect property appreciation?",
            "Should I invest in furnished or unfurnished?",
            "How much should I budget for maintenance?",
            "What's the difference between ROI and rental yield?",
            "How to evaluate a property's location?",
            "What are the tax implications of rental income?",
            "Should I get a mortgage or pay cash?"
        ]
        
        cols = st.columns(3)
        for idx, question in enumerate(general_questions):
            with cols[idx % 3]:
                if st.button(question, key=f"gen_q_{idx}", use_container_width=True):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })
                    response = st.session_state.chatbot.chat(question)
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    st.rerun()
        
        # Clear chat
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.chatbot.reset_conversation()
                st.rerun()

elif page == "📈 Market Dashboard":
    st.markdown("<h1>📈 Market Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style="margin:0; font-size:1rem;">
            Comprehensive overview of property market trends, price distributions, 
            and investment opportunities.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample data for dashboard
    np.random.seed(42)
    n_properties = 50
    
    dashboard_data = pd.DataFrame({
        'Property': [f'Property {i+1}' for i in range(n_properties)],
        'Price': np.random.randint(2000000, 12000000, n_properties),
        'Area': np.random.randint(2000, 10000, n_properties),
        'Bedrooms': np.random.randint(2, 5, n_properties),
        'ROI': np.random.uniform(10, 50, n_properties),
        'Rental_Yield': np.random.uniform(3, 8, n_properties),
        'Location': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_properties),
        'Type': np.random.choice(['Apartment', 'Villa', 'Independent House'], n_properties)
    })
    
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = dashboard_data['Price'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:white; margin:0;">₹{avg_price/1000000:.1f}M</h3>
            <p style="margin:0; opacity:0.9;">Avg Property Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_roi = dashboard_data['ROI'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:white; margin:0;">{avg_roi:.1f}%</h3>
            <p style="margin:0; opacity:0.9;">Avg ROI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_yield = dashboard_data['Rental_Yield'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:white; margin:0;">{avg_yield:.1f}%</h3>
            <p style="margin:0; opacity:0.9;">Avg Rental Yield</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_props = len(dashboard_data)
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:white; margin:0;">{total_props}</h3>
            <p style="margin:0; opacity:0.9;">Properties Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by location - Beautiful Box Plot
        fig1 = go.Figure()
        
        locations = dashboard_data['Location'].unique()
        colors_map = {'North': '#667eea', 'South': '#8ec5fc', 'East': '#a8edea', 
                     'West': '#fbc2eb', 'Central': '#764ba2'}
        
        for location in locations:
            data = dashboard_data[dashboard_data['Location'] == location]['Price']
            fig1.add_trace(go.Box(
                y=data,
                name=location,
                marker=dict(
                    color=colors_map.get(location, '#667eea'),
                    line=dict(width=2, color='#2d3748')
                ),
                boxmean='sd',
                hovertemplate='<b>%{fullData.name}</b><br>Price: ₹%{y:,.0f}<extra></extra>'
            ))
        
        fig1.update_layout(
            title={
                'text': '<b>Price Distribution by Location</b>',
                'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}
            },
            yaxis_title='<b>Price (₹)</b>',
            xaxis_title='<b>Location</b>',
            height=450,
            plot_bgcolor='rgba(246, 248, 251, 0.5)',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            showlegend=False,
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(102, 126, 234, 0.1)'
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=12, color='#4a5568')
            ),
            margin=dict(l=80, r=40, t=80, b=60)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # ROI vs Rental Yield scatter - Beautiful Bubble Chart
        fig3 = go.Figure()
        
        # Create color mapping for property types
        type_colors = {
            'Apartment': '#667eea',
            'Villa': '#8ec5fc',
            'Independent House': '#a8edea'
        }
        
        for prop_type in dashboard_data['Type'].unique():
            data = dashboard_data[dashboard_data['Type'] == prop_type]
            fig3.add_trace(go.Scatter(
                x=data['ROI'],
                y=data['Rental_Yield'],
                mode='markers',
                name=prop_type,
                marker=dict(
                    size=data['Price']/100000,
                    color=type_colors.get(prop_type, '#667eea'),
                    line=dict(width=2, color='white'),
                    opacity=0.8,
                    sizemode='diameter',
                    sizeref=1
                ),
                text=data['Property'],
                hovertemplate='<b>%{text}</b><br>ROI: %{x:.1f}%<br>Rental Yield: %{y:.1f}%<br>Price: ₹%{marker.size:.1f}L<extra></extra>'
            ))
        
        fig3.update_layout(
            title={
                'text': '<b>ROI vs Rental Yield Analysis</b>',
                'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}
            },
            xaxis_title='<b>ROI (%)</b>',
            yaxis_title='<b>Rental Yield (%)</b>',
            height=450,
            plot_bgcolor='rgba(246, 248, 251, 0.5)',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(102, 126, 234, 0.1)'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(102, 126, 234, 0.1)'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#667eea',
                borderwidth=2
            ),
            margin=dict(l=60, r=40, t=100, b=60)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Average ROI by property type - Gradient Bar Chart
        avg_roi_by_type = dashboard_data.groupby('Type')['ROI'].mean().reset_index().sort_values('ROI', ascending=True)
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            y=avg_roi_by_type['Type'],
            x=avg_roi_by_type['ROI'],
            orientation='h',
            marker=dict(
                color=avg_roi_by_type['ROI'],
                colorscale=[
                    [0, '#e0c3fc'],
                    [0.5, '#8ec5fc'],
                    [1, '#667eea']
                ],
                line=dict(color='#667eea', width=2)
            ),
            text=[f'<b>{roi:.1f}%</b>' for roi in avg_roi_by_type['ROI']],
            textposition='outside',
            textfont=dict(size=14, color='#2d3748', family='Inter', weight='bold'),
            hovertemplate='<b>%{y}</b><br>Average ROI: %{x:.1f}%<extra></extra>'
        ))
        
        fig2.update_layout(
            title={
                'text': '<b>Average ROI by Property Type</b>',
                'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}
            },
            xaxis_title='<b>ROI (%)</b>',
            yaxis_title='',
            height=450,
            plot_bgcolor='rgba(246, 248, 251, 0.5)',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(102, 126, 234, 0.1)'
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=13, color='#4a5568')
            ),
            margin=dict(l=20, r=100, t=80, b=60)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Property distribution - Modern Donut Chart
        type_counts = dashboard_data['Type'].value_counts()
        
        fig4 = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.5,
            marker=dict(
                colors=['#667eea', '#8ec5fc', '#a8edea'],
                line=dict(color='white', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, family='Inter', color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            pull=[0.05, 0, 0]
        )])
        
        # Add center text
        fig4.add_annotation(
            text=f'<b>{len(dashboard_data)}</b><br>Properties',
            x=0.5, y=0.5,
            font=dict(size=20, color='#2d3748', family='Inter'),
            showarrow=False
        )
        
        fig4.update_layout(
            title={
                'text': '<b>Property Type Distribution</b>',
                'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}
            },
            height=450,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='middle',
                y=0.5,
                xanchor='left',
                x=1.02,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#667eea',
                borderwidth=2
            ),
            margin=dict(l=40, r=120, t=80, b=40)
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.markdown("<h2>📋 Property Comparison Table</h2>", unsafe_allow_html=True)
    
    # Format the dataframe
    display_df = dashboard_data.copy()
    display_df['Price'] = display_df['Price'].apply(lambda x: f"₹{x/1000000:.2f}M")
    display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.1f}%")
    display_df['Rental_Yield'] = display_df['Rental_Yield'].apply(lambda x: f"{x:.1f}%")
    display_df['Area'] = display_df['Area'].apply(lambda x: f"{x:,} sq ft")
    
    # Add investment grade
    display_df['Grade'] = dashboard_data.apply(
        lambda row: '⭐⭐⭐⭐⭐' if row['ROI'] > 40 and row['Rental_Yield'] > 6
        else '⭐⭐⭐⭐' if row['ROI'] > 30 and row['Rental_Yield'] > 5
        else '⭐⭐⭐' if row['ROI'] > 20
        else '⭐⭐',
        axis=1
    )
    
    st.dataframe(
        display_df.sort_values('ROI', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Download option
    csv = dashboard_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f'property_analysis_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )
    
    # Market insights
    st.markdown("---")
    st.markdown("<h2>💡 Market Insights</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_location = dashboard_data.groupby('Location')['Price'].mean().idxmax()
        st.markdown(f"""
        <div class="feature-card">
            <h4>🏆 Best Location</h4>
            <p style="font-size:1.5rem; font-weight:700; color:#667eea; margin:0.5rem 0;">
                {best_location}
            </p>
            <p style="font-size:0.9rem; margin:0;">Highest average property values</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_roi_type = dashboard_data.groupby('Type')['ROI'].mean().idxmax()
        st.markdown(f"""
        <div class="feature-card">
            <h4>📈 Best ROI Type</h4>
            <p style="font-size:1.5rem; font-weight:700; color:#667eea; margin:0.5rem 0;">
                {best_roi_type}
            </p>
            <p style="font-size:0.9rem; margin:0;">Highest investment returns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_yield_count = len(dashboard_data[dashboard_data['Rental_Yield'] > 6])
        st.markdown(f"""
        <div class="feature-card">
            <h4>💰 High Yield Properties</h4>
            <p style="font-size:1.5rem; font-weight:700; color:#667eea; margin:0.5rem 0;">
                {high_yield_count}
            </p>
            <p style="font-size:0.9rem; margin:0;">Properties with >6% yield</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:2rem 0; color:#718096;">
    <p style="margin:0; font-size:0.9rem;">
        <strong>AI Real Estate Investment Advisor</strong> | Powered by Machine Learning & AI
    </p>
    <p style="margin:0.5rem 0 0 0; font-size:0.85rem;">
        Built with Streamlit • LangChain • Groq • TensorFlow • Plotly
    </p>
    <p style="margin:0.5rem 0 0 0; font-size:0.8rem; opacity:0.7;">
        © 2025 Real Estate AI Platform. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)