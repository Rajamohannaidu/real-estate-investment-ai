# app/streamlit_app.py - Enhanced Modern Design (Merged Version)

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
from pathlib import Path

# Add parent directory to path - using both methods for compatibility
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import RealEstateDataPreprocessor
from src.predictive_models import RealEstatePredictiveModels
from src.investment_analytics import InvestmentAnalytics
from src.explainability import ModelExplainability

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
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        border: 2px solid #e2e8f0;
        border-bottom: none;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        border-radius: 10px;
        padding: 1rem;
        font-weight: 600;
        border: 2px solid #e2e8f0;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #ffffff 0%, #f6f8fb 100%);
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
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = RealEstateDataPreprocessor()

if 'analytics' not in st.session_state:
    st.session_state.analytics = InvestmentAnalytics()

if 'models' not in st.session_state:
    st.session_state.models = None
    st.session_state.model_state = None

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load pre-trained models on first run
if st.session_state.models is None:
    with st.spinner("🔄 Loading AI models..."):
        st.session_state.models, st.session_state.model_state = load_pretrained_models()

# Title and Introduction
st.markdown("<h1>🏠 AI Real Estate Investment Advisor</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <p style="font-size:1.1rem; margin:0; line-height:1.6;">
        Welcome to your intelligent real estate investment companion! Leverage cutting-edge AI and machine learning 
        to make data-driven investment decisions. Analyze properties, predict prices, calculate ROI, and get 
        personalized investment recommendations.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:white;'>🎯 Navigation</h2>", unsafe_allow_html=True)
    
    page = st.radio(
        "",
        ["🏠 Home", "📊 Data Upload & Analysis", "🤖 Price Prediction", 
         "💰 Investment Analysis", "🔍 Model Insights", "💬 AI Chatbot"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model Status
    st.markdown("<h3 style='color:white;'>🤖 Model Status</h3>", unsafe_allow_html=True)
    
    if st.session_state.models:
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.1); padding:1rem; border-radius:10px; margin:0.5rem 0;'>
            <p style='margin:0; color:#4ade80;'>✅ Models Loaded</p>
            <p style='margin:0.5rem 0 0 0; font-size:0.85rem; color:#e0e7ff;'>
                {len(st.session_state.models.models)} models ready
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.1); padding:1rem; border-radius:10px; margin:0.5rem 0;'>
            <p style='margin:0; color:#fbbf24;'>⚠️ No Models Found</p>
            <p style='margin:0.5rem 0 0 0; font-size:0.85rem; color:#e0e7ff;'>
                Please train models first
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.data_loaded:
        st.markdown("""
        <div style='background:rgba(255,255,255,0.1); padding:1rem; border-radius:10px; margin:0.5rem 0;'>
            <p style='margin:0; color:#4ade80;'>✅ Data Loaded</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("<h3 style='color:white;'>📈 Quick Stats</h3>", unsafe_allow_html=True)
    
    if st.session_state.data_loaded and st.session_state.preprocessor:
        df = st.session_state.preprocessor.data
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.1); padding:1rem; border-radius:10px;'>
            <p style='margin:0; color:#e0e7ff; font-size:0.9rem;'>
                📍 Properties: <strong>{len(df)}</strong><br>
                📊 Features: <strong>{len(df.columns)}</strong><br>
                🏢 Types: <strong>{df['Type'].nunique() if 'Type' in df.columns else 'N/A'}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# Main Content
if page == "🏠 Home":
    st.markdown("<h2>🎯 Platform Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Data Analysis</h3>
            <p style="margin:0.5rem 0; line-height:1.6;">
                Upload and analyze property data with advanced preprocessing, 
                missing value handling, and feature engineering.
            </p>
            <ul style="margin:0.5rem 0; padding-left:1.5rem; line-height:1.8;">
                <li>Automated data cleaning</li>
                <li>Feature engineering</li>
                <li>Statistical analysis</li>
                <li>Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Predictions</h3>
            <p style="margin:0.5rem 0; line-height:1.6;">
                Leverage multiple machine learning models for accurate 
                property price predictions.
            </p>
            <ul style="margin:0.5rem 0; padding-left:1.5rem; line-height:1.8;">
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
                <li>XGBoost</li>
                <li>Neural Networks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>💰 Investment Analytics</h3>
            <p style="margin:0.5rem 0; line-height:1.6;">
                Comprehensive investment analysis with ROI calculations, 
                risk assessment, and portfolio optimization.
            </p>
            <ul style="margin:0.5rem 0; padding-left:1.5rem; line-height:1.8;">
                <li>ROI calculations</li>
                <li>Risk assessment</li>
                <li>Market comparison</li>
                <li>Portfolio insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2>🚀 Getting Started</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📝 Step 1: Upload Your Data</h3>
            <p style="margin:0.5rem 0; line-height:1.6;">
                Navigate to the <strong>Data Upload & Analysis</strong> page and upload your 
                real estate dataset in CSV format. The system will automatically:
            </p>
            <ul style="margin:0.5rem 0; padding-left:1.5rem; line-height:1.8;">
                <li>Detect data types</li>
                <li>Handle missing values</li>
                <li>Engineer relevant features</li>
                <li>Generate insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Step 2: Analyze & Predict</h3>
            <p style="margin:0.5rem 0; line-height:1.6;">
                Use the <strong>Price Prediction</strong> and <strong>Investment Analysis</strong> 
                pages to:
            </p>
            <ul style="margin:0.5rem 0; padding-left:1.5rem; line-height:1.8;">
                <li>Predict property prices</li>
                <li>Calculate investment returns</li>
                <li>Assess risk factors</li>
                <li>Compare opportunities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<h2>📚 Sample Insights</h2>", unsafe_allow_html=True)
    
    # Sample dashboard with mock data
    sample_data = {
        'Property': ['Luxury Villa', 'Modern Apartment', 'Commercial Space', 'Beach House'],
        'Location': ['Downtown', 'Suburbs', 'Business District', 'Coastal'],
        'Type': ['Villa', 'Apartment', 'Commercial', 'Villa'],
        'Area': [3500, 1200, 2000, 4000],
        'Price': [15000000, 5000000, 12000000, 25000000],
        'ROI': [35.5, 28.3, 42.1, 31.8],
        'Rental_Yield': [6.2, 5.8, 7.5, 5.5]
    }
    
    dashboard_data = pd.DataFrame(sample_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("💰 Avg Property Price", f"₹{dashboard_data['Price'].mean()/10000000:.1f}Cr", "+12.5%"),
        ("📈 Avg ROI", f"{dashboard_data['ROI'].mean():.1f}%", "+5.2%"),
        ("🏘️ Properties", str(len(dashboard_data)), "+3"),
        ("⭐ Avg Yield", f"{dashboard_data['Rental_Yield'].mean():.1f}%", "+0.8%")
    ]
    
    for col, (label, value, delta) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; font-size:0.9rem; opacity:0.9;">{label}</h4>
                <p style="font-size:2rem; font-weight:700; margin:0.5rem 0;">{value}</p>
                <p style="margin:0; font-size:0.85rem; opacity:0.8;">
                    <span style="color:#4ade80;">↗ {delta}</span> vs last month
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by location - Modern Bar Chart
        avg_price_by_loc = dashboard_data.groupby('Location')['Price'].mean().reset_index().sort_values('Price', ascending=False)
        
        fig1 = go.Figure()
        
        fig1.add_trace(go.Bar(
            x=avg_price_by_loc['Location'],
            y=avg_price_by_loc['Price']/1000000,
            marker=dict(
                color=avg_price_by_loc['Price']/1000000,
                colorscale='Blues',
                line=dict(color='#667eea', width=2)
            ),
            text=[f'₹{p/10:.1f}Cr' for p in avg_price_by_loc['Price']/1000000],
            textposition='outside',
            textfont=dict(size=13, color='#2d3748', family='Inter', weight='bold'),
            hovertemplate='<b>%{x}</b><br>Avg Price: ₹%{y:.1f}M<extra></extra>'
        ))
        
        fig1.update_layout(
            title={
                'text': '<b>Average Price by Location</b>',
                'font': {'size': 18, 'color': '#2d3748', 'family': 'Inter'}
            },
            xaxis_title='<b>Location</b>',
            yaxis_title='<b>Price (₹ Million)</b>',
            height=450,
            plot_bgcolor='rgba(246, 248, 251, 0.5)',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=12, color='#4a5568')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(102, 126, 234, 0.1)'
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # ROI vs Rental Yield Scatter Plot
        data = dashboard_data.copy()
        
        fig3 = go.Figure()
        
        for ptype in data['Type'].unique():
            type_data = data[data['Type'] == ptype]
            fig3.add_trace(go.Scatter(
                x=type_data['ROI'],
                y=type_data['Rental_Yield'],
                mode='markers',
                name=ptype,
                marker=dict(
                    size=type_data['Price']/500000,
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

elif page == "📊 Data Upload & Analysis":
    st.markdown("<h2>📊 Data Upload & Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <p style="margin:0; line-height:1.6;">
            Upload your real estate dataset to begin analysis. The system supports CSV files 
            with property information including features like location, size, price, amenities, etc.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing real estate data"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Successfully loaded dataset with {len(df)} properties and {len(df.columns)} features")
            
            # Store data in preprocessor
            st.session_state.preprocessor.data = df
            st.session_state.data_loaded = True
            
            # Data Overview
            st.markdown("---")
            st.markdown("<h3>📋 Data Overview</h3>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Properties", len(df))
            with col2:
                st.metric("Total Features", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
            
            # Display first few rows
            st.markdown("<h3>🔍 Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(10), use_container_width=True)
            
            # Data Info
            st.markdown("<h3>ℹ️ Dataset Information</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Numeric Columns:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                st.write(numeric_cols)
                
            with col2:
                st.markdown("**Categorical Columns:**")
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                st.write(cat_cols)
            
            # Statistical Summary
            st.markdown("<h3>📊 Statistical Summary</h3>", unsafe_allow_html=True)
            st.dataframe(df.describe(), use_container_width=True)
            
            # Missing Values Analysis
            if df.isnull().sum().sum() > 0:
                st.markdown("<h3>⚠️ Missing Values</h3>", unsafe_allow_html=True)
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isnull().sum().values,
                    'Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            
            # Data Preprocessing
            st.markdown("---")
            st.markdown("<h3>🔧 Data Preprocessing</h3>", unsafe_allow_html=True)
            
            if st.button("🚀 Start Preprocessing", type="primary"):
                with st.spinner("Processing data..."):
                    try:
                        # Clean data
                        processed_df = st.session_state.preprocessor.clean_data(df.copy())
                        
                        # Feature engineering
                        processed_df = st.session_state.preprocessor.feature_engineering(processed_df)
                        
                        # Encode categorical columns
                        categorical_cols = [col for col in processed_df.columns 
                                          if processed_df[col].dtype == 'object' and col.lower() != 'price']
                        if categorical_cols:
                            processed_df = st.session_state.preprocessor.encode_categorical(processed_df, categorical_cols)
                        
                        st.success("✅ Data preprocessing completed!")
                        
                        # Show processed data
                        st.markdown("<h4>Processed Data Preview</h4>", unsafe_allow_html=True)
                        st.dataframe(processed_df.head(), use_container_width=True)
                        
                        # Show preprocessing summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Original Features", len(df.columns))
                        with col2:
                            st.metric("Processed Features", len(processed_df.columns))
                        with col3:
                            st.metric("New Features", len(processed_df.columns) - len(df.columns))
                        
                        # Download processed data
                        csv = processed_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Processed Data",
                            data=csv,
                            file_name=f'processed_data_{datetime.now().strftime("%Y%m%d")}.csv',
                            mime='text/csv',
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during preprocessing: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis")

elif page == "🤖 Price Prediction":
    st.markdown("<h2>🤖 Price Prediction</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Analysis' section")
    elif not st.session_state.models:
        st.warning("⚠️ No trained models available. Please train models first.")
    else:
        st.markdown("""
        <div class="info-card">
            <p style="margin:0; line-height:1.6;">
                Use our AI-powered models to predict property prices based on various features. 
                Enter the property details below to get instant price predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if we have feature names from trained model
        if not st.session_state.models.feature_names:
            st.error("⚠️ Model feature information not available. Please retrain the models.")
        else:
            # Input form - use ORIGINAL features (before preprocessing)
            st.markdown("<h3>🏘️ Property Details</h3>", unsafe_allow_html=True)
            
            # Get the original data to understand feature ranges
            df = st.session_state.preprocessor.data
            
            # Identify original numeric and categorical features
            # Exclude engineered features (those created during feature_engineering)
            engineered_suffixes = ['_category', '_ratio', '_score', '_per_', '_age', '_density']
            
            original_numeric = []
            original_categorical = []
            
            for col in df.columns:
                # Skip price column and engineered features
                if col.lower() in ['price', 'target']:
                    continue
                    
                is_engineered = any(suffix in col.lower() for suffix in engineered_suffixes)
                if is_engineered:
                    continue
                    
                if df[col].dtype in ['int64', 'float64']:
                    original_numeric.append(col)
                elif df[col].dtype == 'object':
                    original_categorical.append(col)
            
            # Create input form
            inputs = {}
            
            st.markdown("**📊 Numeric Features**")
            
            # Dynamic columns based on number of features
            num_cols = min(3, len(original_numeric)) if original_numeric else 1
            cols = st.columns(num_cols)
            
            for idx, feature in enumerate(original_numeric):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    # Use median as default, provide reasonable min/max
                    median_val = float(df[feature].median())
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    
                    inputs[feature] = st.number_input(
                        feature.replace('_', ' ').title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=median_val,
                        help=f"Enter {feature} (range: {min_val:.0f} - {max_val:.0f})"
                    )
            
            if original_categorical:
                st.markdown("---")
                st.markdown("**🏷️ Categorical Features**")
                
                num_cols_cat = min(3, len(original_categorical))
                cols_cat = st.columns(num_cols_cat)
                
                for idx, feature in enumerate(original_categorical):
                    col_idx = idx % num_cols_cat
                    with cols_cat[col_idx]:
                        unique_vals = df[feature].unique().tolist()
                        inputs[feature] = st.selectbox(
                            feature.replace('_', ' ').title(),
                            options=unique_vals,
                            help=f"Select {feature}"
                        )
            
            st.markdown("---")
            
            if st.button("🎯 Predict Price", type="primary", use_container_width=True):
                with st.spinner("Calculating prediction..."):
                    try:
                        # Prepare input data
                        input_df = pd.DataFrame([inputs])
                        
                        # Apply the same preprocessing pipeline as training data
                        # 1. Feature engineering (creates new features)
                        input_df_processed = st.session_state.preprocessor.feature_engineering(input_df.copy())
                        
                        # 2. Encode categorical features
                        categorical_cols = [col for col in input_df_processed.columns 
                                          if input_df_processed[col].dtype == 'object']
                        if categorical_cols:
                            input_df_processed = st.session_state.preprocessor.encode_categorical(
                                input_df_processed, categorical_cols
                            )
                        
                        # 3. Ensure all required features are present and in correct order
                        expected_features = st.session_state.models.feature_names
                        
                        # Add missing features with 0 (this handles any features not in input)
                        for feature in expected_features:
                            if feature not in input_df_processed.columns:
                                input_df_processed[feature] = 0
                        
                        # Select only the features the model expects, in the correct order
                        input_df_final = input_df_processed[expected_features]
                        
                        # Make prediction using best model
                        prediction = st.session_state.models.best_model.predict(input_df_final)[0]
                        
                        # Display prediction
                        st.markdown("---")
                        st.markdown("<h3>📊 Prediction Results</h3>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin:0; opacity:0.9;">Predicted Price</h4>
                                <p style="font-size:2.5rem; font-weight:700; margin:1rem 0;">
                                    ₹{prediction/10000000:.2f}Cr
                                </p>
                                <p style="margin:0; opacity:0.8;">₹{prediction:,.0f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Calculate price range (±10%)
                            lower = prediction * 0.9
                            upper = prediction * 1.1
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin:0; opacity:0.9;">Price Range</h4>
                                <p style="font-size:1.5rem; font-weight:700; margin:1rem 0;">
                                    ₹{lower/10000000:.2f}Cr - ₹{upper/10000000:.2f}Cr
                                </p>
                                <p style="margin:0; opacity:0.8;">±10% confidence interval</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            # Model confidence
                            if st.session_state.models.best_model_name in st.session_state.models.model_metrics:
                                r2 = st.session_state.models.model_metrics[st.session_state.models.best_model_name]['r2_score']
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4 style="margin:0; opacity:0.9;">Model Accuracy</h4>
                                    <p style="font-size:2.5rem; font-weight:700; margin:1rem 0;">
                                        {r2*100:.1f}%
                                    </p>
                                    <p style="margin:0; opacity:0.8;">R² Score</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show input summary
                        st.markdown("---")
                        st.markdown("<h3>📋 Input Summary</h3>", unsafe_allow_html=True)
                        
                        input_summary = pd.DataFrame([inputs]).T
                        input_summary.columns = ['Value']
                        input_summary.index.name = 'Feature'
                        st.dataframe(input_summary, use_container_width=True)
                        
                        st.success("✅ Prediction completed successfully!")
                        
                        # Show model used
                        st.info(f"🤖 Prediction made using: **{st.session_state.models.best_model_name.replace('_', ' ').title()}**")
                        
                    except Exception as e:
                        st.error(f"❌ Error making prediction: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
                            st.write("**Input data shape:**", input_df.shape if 'input_df' in locals() else "N/A")
                            st.write("**Processed data shape:**", input_df_processed.shape if 'input_df_processed' in locals() else "N/A")
                            st.write("**Expected features:**", len(expected_features) if 'expected_features' in locals() else "N/A")
                            if 'input_df_processed' in locals():
                                st.write("**Available features:**", input_df_processed.columns.tolist())

elif page == "💰 Investment Analysis":
    st.markdown("<h2>💰 Investment Analysis</h2>", unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Analysis' section")
    else:
        st.markdown("""
        <div class="info-card">
            <p style="margin:0; line-height:1.6;">
                Analyze investment potential with ROI calculations, risk assessment, and market comparisons.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.preprocessor.data
        
        # Investment parameters
        st.markdown("<h3>📊 Investment Parameters</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            property_price = st.number_input(
                "Property Price (₹)",
                value=5000000,
                step=100000,
                help="Enter the property price"
            )
        
        with col2:
            monthly_rent = st.number_input(
                "Expected Monthly Rent (₹)",
                value=25000,
                step=1000,
                help="Enter expected monthly rental income"
            )
        
        with col3:
            holding_period = st.number_input(
                "Holding Period (years)",
                value=5,
                min_value=1,
                max_value=30,
                help="Investment holding period"
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            appreciation_rate = st.slider(
                "Annual Appreciation Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=8.0,
                step=0.5,
                help="Expected annual property appreciation"
            )
        
        with col2:
            maintenance_cost = st.slider(
                "Annual Maintenance (%)",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Annual maintenance as % of property value"
            )
        
        with col3:
            tax_rate = st.slider(
                "Tax Rate (%)",
                min_value=0.0,
                max_value=30.0,
                value=20.0,
                step=1.0,
                help="Applicable tax rate"
            )
        
        if st.button("📈 Calculate Investment Returns", type="primary"):
            with st.spinner("Analyzing investment..."):
                # Calculate metrics
                annual_rent = monthly_rent * 12
                rental_yield = (annual_rent / property_price) * 100
                
                # Future value
                future_value = property_price * ((1 + appreciation_rate/100) ** holding_period)
                capital_gain = future_value - property_price
                
                # Total rental income
                total_rental_income = annual_rent * holding_period
                
                # Maintenance costs
                total_maintenance = (property_price * maintenance_cost/100) * holding_period
                
                # Net profit
                net_profit = capital_gain + total_rental_income - total_maintenance
                roi = (net_profit / property_price) * 100
                
                # Display results
                st.markdown("---")
                st.markdown("<h3>📊 Investment Analysis Results</h3>", unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0; opacity:0.9;">Rental Yield</h4>
                        <p style="font-size:2rem; font-weight:700; margin:1rem 0;">
                            {rental_yield:.2f}%
                        </p>
                        <p style="margin:0; opacity:0.8;">Annual Return</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0; opacity:0.9;">Total ROI</h4>
                        <p style="font-size:2rem; font-weight:700; margin:1rem 0;">
                            {roi:.1f}%
                        </p>
                        <p style="margin:0; opacity:0.8;">{holding_period} years</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0; opacity:0.9;">Future Value</h4>
                        <p style="font-size:2rem; font-weight:700; margin:1rem 0;">
                            ₹{future_value/10000000:.2f}Cr
                        </p>
                        <p style="margin:0; opacity:0.8;">After {holding_period} years</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin:0; opacity:0.9;">Net Profit</h4>
                        <p style="font-size:2rem; font-weight:700; margin:1rem 0;">
                            ₹{net_profit/10000000:.2f}Cr
                        </p>
                        <p style="margin:0; opacity:0.8;">Total Returns</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed breakdown
                st.markdown("<h3>💵 Financial Breakdown</h3>", unsafe_allow_html=True)
                
                breakdown_df = pd.DataFrame({
                    'Component': [
                        'Initial Investment',
                        'Capital Appreciation',
                        'Total Rental Income',
                        'Maintenance Costs',
                        'Net Profit',
                        'ROI'
                    ],
                    'Amount': [
                        f"₹{property_price/10000000:.2f}Cr",
                        f"₹{capital_gain/10000000:.2f}Cr",
                        f"₹{total_rental_income/10000000:.2f}Cr",
                        f"-₹{total_maintenance/10000000:.2f}Cr",
                        f"₹{net_profit/10000000:.2f}Cr",
                        f"{roi:.1f}%"
                    ]
                })
                
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                
                st.success("✅ Investment analysis completed!")

elif page == "🔍 Model Insights":
    st.markdown("<h2>🔍 Model Insights</h2>", unsafe_allow_html=True)
    
    if not st.session_state.models:
        st.warning("⚠️ No trained models available.")
    else:
        st.markdown("""
        <div class="info-card">
            <p style="margin:0; line-height:1.6;">
                Explore model performance metrics, feature importance, and explainability insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Performance
        st.markdown("<h3>📊 Model Performance Comparison</h3>", unsafe_allow_html=True)
        
        metrics_df = pd.DataFrame(st.session_state.models.model_metrics).T
        metrics_df = metrics_df.reset_index()
        metrics_df.columns = ['Model', 'R² Score', 'RMSE', 'MAE', 'MAPE']
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Best model highlight
        best_idx = metrics_df['R² Score'].idxmax()
        best_model = metrics_df.loc[best_idx, 'Model']
        best_r2 = metrics_df.loc[best_idx, 'R² Score']
        
        st.markdown(f"""
        <div class="info-card">
            <h4>🏆 Best Performing Model</h4>
            <p style="margin:0.5rem 0; font-size:1.2rem;">
                <strong>{best_model}</strong> with R² Score of <strong>{best_r2:.4f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df['R² Score'],
                    marker=dict(
                        color=metrics_df['R² Score'],
                        colorscale='Blues',
                        line=dict(color='#667eea', width=2)
                    ),
                    text=metrics_df['R² Score'].round(4),
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="R² Score Comparison",
                xaxis_title="Model",
                yaxis_title="R² Score",
                height=400,
                plot_bgcolor='rgba(246, 248, 251, 0.5)',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics_df['Model'],
                    y=metrics_df['RMSE'],
                    marker=dict(
                        color=metrics_df['RMSE'],
                        colorscale='Reds',
                        reversescale=True,
                        line=dict(color='#764ba2', width=2)
                    ),
                    text=metrics_df['RMSE'].round(0),
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="RMSE Comparison (Lower is Better)",
                xaxis_title="Model",
                yaxis_title="RMSE",
                height=400,
                plot_bgcolor='rgba(246, 248, 251, 0.5)',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

elif page == "💬 AI Chatbot":
    st.markdown("<h2>💬 AI Real Estate Assistant</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <p style="margin:0; line-height:1.6;">
            Ask questions about your real estate data, get investment insights, and receive 
            personalized recommendations based on your uploaded properties and market analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Context-based chatbot implementation
    def get_data_context():
        """Generate context from loaded data"""
        context = {
            'has_data': st.session_state.data_loaded,
            'has_models': st.session_state.models is not None,
            'data_summary': None,
            'model_info': None
        }
        
        if st.session_state.data_loaded and st.session_state.preprocessor.data is not None:
            df = st.session_state.preprocessor.data
            
            # Get numeric columns (excluding price)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            price_col = None
            for col in ['Price', 'price', 'target', 'Target']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col and price_col in numeric_cols:
                numeric_cols.remove(price_col)
            
            context['data_summary'] = {
                'total_properties': len(df),
                'columns': df.columns.tolist(),
                'numeric_features': numeric_cols,
                'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            }
            
            # Add statistical summary
            if price_col:
                context['data_summary']['price_stats'] = {
                    'mean': float(df[price_col].mean()),
                    'median': float(df[price_col].median()),
                    'min': float(df[price_col].min()),
                    'max': float(df[price_col].max()),
                    'std': float(df[price_col].std())
                }
            
            # Add categorical breakdowns
            cat_cols = df.select_dtypes(include=['object']).columns
            context['data_summary']['categorical_distributions'] = {}
            for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                context['data_summary']['categorical_distributions'][col] = df[col].value_counts().to_dict()
        
        if st.session_state.models and st.session_state.models.model_metrics:
            context['model_info'] = {
                'available_models': list(st.session_state.models.models.keys()),
                'best_model': st.session_state.models.best_model_name,
                'metrics': st.session_state.models.model_metrics
            }
        
        return context
    
    def generate_response(user_question, context):
        """Generate contextual response based on question and data"""
        question_lower = user_question.lower()
        
        # Check if data is loaded
        if not context['has_data']:
            return "I don't have any data loaded yet. Please upload your real estate data in the 'Data Upload & Analysis' section first, and I'll be able to provide insights based on your specific properties."
        
        data_summary = context['data_summary']
        
        # Property count questions
        if any(word in question_lower for word in ['how many', 'number of', 'total properties', 'count']):
            response = f"Based on your uploaded data, you have **{data_summary['total_properties']} properties** in your dataset."
            
            if data_summary['categorical_distributions']:
                first_cat = list(data_summary['categorical_distributions'].keys())[0]
                dist = data_summary['categorical_distributions'][first_cat]
                response += f"\n\nBreakdown by {first_cat}:\n"
                for key, value in dist.items():
                    response += f"- {key}: {value} properties\n"
            
            return response
        
        # Price-related questions
        if any(word in question_lower for word in ['price', 'cost', 'expensive', 'cheap', 'affordable']):
            if data_summary['price_stats']:
                stats = data_summary['price_stats']
                response = f"""**Price Analysis of Your Properties:**

- **Average Price:** ₹{stats['mean']/10000000:.2f} Crore (₹{stats['mean']:,.0f})
- **Median Price:** ₹{stats['median']/10000000:.2f} Crore (₹{stats['median']:,.0f})
- **Price Range:** ₹{stats['min']/10000000:.2f}Cr to ₹{stats['max']/10000000:.2f}Cr
- **Standard Deviation:** ₹{stats['std']/10000000:.2f} Crore

**Insights:**
"""
                if stats['mean'] > stats['median']:
                    response += "- The average price is higher than the median, suggesting some high-value properties are pulling the average up.\n"
                
                price_range = stats['max'] - stats['min']
                if price_range > stats['mean'] * 2:
                    response += "- There's significant price diversity in your portfolio, which is good for risk distribution.\n"
                
                if 'most expensive' in question_lower or 'highest' in question_lower:
                    response += f"- The most expensive property is priced at ₹{stats['max']/10000000:.2f} Crore.\n"
                
                if 'cheapest' in question_lower or 'lowest' in question_lower:
                    response += f"- The most affordable property is priced at ₹{stats['min']/10000000:.2f} Crore.\n"
                
                return response
        
        # Feature-related questions
        if any(word in question_lower for word in ['features', 'columns', 'attributes', 'what data']):
            response = f"**Your Dataset Contains {len(data_summary['columns'])} Features:**\n\n"
            
            if data_summary['numeric_features']:
                response += "**Numeric Features:**\n"
                for feat in data_summary['numeric_features'][:10]:
                    response += f"- {feat}\n"
                if len(data_summary['numeric_features']) > 10:
                    response += f"- ... and {len(data_summary['numeric_features']) - 10} more\n"
                response += "\n"
            
            if data_summary['categorical_features']:
                response += "**Categorical Features:**\n"
                for feat in data_summary['categorical_features'][:10]:
                    response += f"- {feat}\n"
                if len(data_summary['categorical_features']) > 10:
                    response += f"- ... and {len(data_summary['categorical_features']) - 10} more\n"
            
            return response
        
        # Model-related questions
        if any(word in question_lower for word in ['model', 'accuracy', 'prediction', 'ai', 'machine learning']):
            if context['has_models']:
                model_info = context['model_info']
                response = f"**AI Models Information:**\n\n"
                response += f"**Best Performing Model:** {model_info['best_model'].replace('_', ' ').title()}\n\n"
                
                response += "**Model Performance:**\n"
                for model_name, metrics in model_info['metrics'].items():
                    response += f"\n**{model_name.replace('_', ' ').title()}:**\n"
                    response += f"- R² Score: {metrics['r2_score']:.4f} ({metrics['r2_score']*100:.2f}% accuracy)\n"
                    response += f"- RMSE: ₹{metrics['rmse']:,.0f}\n"
                    response += f"- MAE: ₹{metrics['mae']:,.0f}\n"
                
                response += "\n**What this means:**\n"
                best_r2 = model_info['metrics'][model_info['best_model']]['r2_score']
                if best_r2 > 0.9:
                    response += "- Your models are highly accurate (>90%), making them very reliable for predictions.\n"
                elif best_r2 > 0.8:
                    response += "- Your models have good accuracy (80-90%), suitable for investment decisions.\n"
                else:
                    response += "- Your models have moderate accuracy. Consider collecting more data or additional features.\n"
                
                return response
            else:
                return "No trained models are currently available. You can train models in the 'Model Insights' section to enable AI-powered predictions."
        
        # Investment advice
        if any(word in question_lower for word in ['invest', 'buy', 'recommend', 'suggest', 'should i', 'advice']):
            if data_summary['price_stats']:
                stats = data_summary['price_stats']
                response = "**Investment Advice Based on Your Data:**\n\n"
                
                response += "**General Recommendations:**\n"
                response += "1. **Diversification:** Look for properties across different price ranges and types.\n"
                response += "2. **Market Analysis:** "
                
                if stats['mean'] > 7000000:
                    response += "Your portfolio focuses on premium properties. Consider some mid-range options for balance.\n"
                elif stats['mean'] < 3000000:
                    response += "Your portfolio is in the affordable segment. This offers good rental yield potential.\n"
                else:
                    response += "Your portfolio is well-balanced in the mid-range segment.\n"
                
                response += "3. **Risk Assessment:** "
                if stats['std'] > stats['mean'] * 0.5:
                    response += "High price variation suggests diverse risk levels. Good for portfolio balance.\n"
                else:
                    response += "Relatively consistent pricing suggests stable market segment.\n"
                
                response += "\n**Next Steps:**\n"
                response += "- Use the Price Prediction feature to estimate property values\n"
                response += "- Check Investment Analysis for ROI calculations\n"
                response += "- Compare properties in the dashboard to find best opportunities\n"
                
                return response
        
        # ROI questions
        if any(word in question_lower for word in ['roi', 'return', 'profit', 'gain']):
            return """**Understanding ROI in Real Estate:**

**Formula:** ROI = (Net Profit / Investment Cost) × 100

**Components to Consider:**
1. **Purchase Price** - Initial investment
2. **Rental Income** - Monthly/annual rent
3. **Appreciation** - Property value increase over time
4. **Maintenance Costs** - Repairs, property tax, etc.
5. **Rental Yield** - Annual rent / Property value

**Good ROI Benchmarks:**
- **Rental Yield:** 4-8% is considered good
- **Total ROI (5 years):** 30-50% is solid
- **Annual Appreciation:** 5-10% is healthy

**Tips for Better ROI:**
- Choose properties in growing areas
- Maintain good tenant relationships
- Keep maintenance costs under control
- Consider tax benefits

Use the **Investment Analysis** page to calculate ROI for specific properties!
"""
        
        # Location questions
        if any(word in question_lower for word in ['location', 'area', 'where', 'best location']):
            if data_summary['categorical_distributions']:
                for col in data_summary['categorical_distributions'].keys():
                    if 'location' in col.lower() or 'area' in col.lower() or 'city' in col.lower():
                        dist = data_summary['categorical_distributions'][col]
                        response = f"**Location Distribution in Your Data:**\n\n"
                        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                        for location, count in sorted_dist:
                            response += f"- **{location}:** {count} properties\n"
                        
                        response += f"\n**Most Common Location:** {sorted_dist[0][0]} with {sorted_dist[0][1]} properties\n"
                        return response
            
            return "I can provide location-specific insights if your data includes location/area information. Make sure your dataset has a location or area column."
        
        # General help
        if any(word in question_lower for word in ['help', 'what can you', 'how to use', 'guide']):
            return """**I can help you with:**

🏠 **Property Analysis:**
- Ask about property counts, distributions
- Get price statistics and insights
- Understand your dataset features

📊 **Investment Insights:**
- ROI calculations and benchmarks
- Investment recommendations
- Market analysis based on your data

🤖 **AI Models:**
- Model performance and accuracy
- Prediction capabilities
- Model comparison

💡 **Example Questions:**
- "How many properties do I have?"
- "What's the average price?"
- "Tell me about my dataset features"
- "How accurate are the models?"
- "Should I invest in property X?"
- "What's a good ROI?"

Just ask anything about your real estate data and investments!
"""
        
        # Market trends
        if any(word in question_lower for word in ['trend', 'market', 'growth', 'future']):
            return """**Real Estate Market Insights:**

**Current Trends (2024-2025):**
1. **Digital Transformation:** PropTech adoption is increasing
2. **Sustainability:** Green buildings command premium prices
3. **Work from Home:** Suburban areas gaining popularity
4. **Smart Homes:** IoT integration is becoming standard

**Investment Strategies:**
- **Long-term:** Focus on location, quality, and infrastructure
- **Short-term:** Look for undervalued properties in growing areas
- **Rental:** Target areas with strong job markets

**Key Factors to Watch:**
- Interest rate changes
- Infrastructure development
- Government policies (tax, subsidies)
- Local economic growth

Use the **Price Prediction** feature to estimate future values based on current trends!
"""
        
        # Default response with context
        response = f"""I'm your AI Real Estate Assistant with access to your data ({data_summary['total_properties']} properties).

**I can help you with:**
- Property statistics and analysis
- Price insights and comparisons
- Investment recommendations
- ROI calculations
- Model performance information

**Try asking:**
- "What's the average price in my portfolio?"
- "How many properties do I have?"
- "Tell me about the models"
- "Should I invest in real estate?"
- "What's a good ROI?"

What would you like to know about your properties?
"""
        return response
    
    # Chat interface
    st.markdown("<h3>💭 Chat with Your AI Assistant</h3>", unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="background:#e0e7ff; padding:1rem; border-radius:10px; margin:0.5rem 0;">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#f6f8fb; padding:1rem; border-radius:10px; margin:0.5rem 0; border-left:4px solid #667eea;">
                    <strong>🤖 AI Assistant:</strong><br>{message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # User input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What's the average price of properties in my dataset?",
            label_visibility="collapsed",
            key="chat_input"
        )
    
    with col2:
        send_button = st.button("Send 📤", type="primary", use_container_width=True)
    
    col_clear1, col_clear2 = st.columns([5, 1])
    with col_clear2:
        if st.button("Clear 🗑️", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if send_button and user_question:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        # Generate response
        with st.spinner("Thinking..."):
            context = get_data_context()
            response = generate_response(user_question, context)
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
        
        st.rerun()
    
    # Sample questions
    st.markdown("---")
    st.markdown("<h3>💡 Sample Questions</h3>", unsafe_allow_html=True)
    
    sample_categories = {
        "📊 Data Analysis": [
            "How many properties do I have?",
            "What's the average price?",
            "Tell me about my dataset features"
        ],
        "💰 Investment": [
            "Should I invest in real estate?",
            "What's a good ROI?",
            "How do I calculate rental yield?"
        ],
        "🤖 AI Models": [
            "How accurate are the predictions?",
            "Which model is best?",
            "Tell me about the models"
        ],
        "🏠 Properties": [
            "What's the most expensive property?",
            "Show me price distribution",
            "What locations are in my data?"
        ]
    }
    
    for category, questions in sample_categories.items():
        with st.expander(category):
            for q in questions:
                if st.button(q, key=f"sample_{q}", use_container_width=True):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': q
                    })
                    
                    context = get_data_context()
                    response = generate_response(q, context)
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    st.rerun()
    
    # Data context indicator
    if st.session_state.data_loaded:
        st.markdown("---")
        context = get_data_context()
        st.markdown(f"""
        <div style="background:rgba(102, 126, 234, 0.1); padding:1rem; border-radius:10px; border-left:4px solid #667eea;">
            <strong>📊 Context Available:</strong><br>
            • {context['data_summary']['total_properties']} properties loaded<br>
            • {len(context['data_summary']['columns'])} features available<br>
            {'• ' + str(len(context['model_info']['available_models'])) + ' AI models ready' if context['has_models'] else '• No models trained yet'}
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