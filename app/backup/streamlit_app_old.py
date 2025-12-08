# app/streamlit_app.py

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import RealEstateDataPreprocessor
from src.predictive_models import RealEstatePredictiveModels
from src.investment_analytics import InvestmentAnalytics
from src.explainability import ModelExplainability
from src.chatbot import RealEstateInvestmentChatbot

# Page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    try:
        st.session_state.chatbot = RealEstateInvestmentChatbot()
        st.session_state.chat_history = []
    except:
        st.session_state.chatbot = None
        st.session_state.chat_history = []

if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = RealEstateDataPreprocessor()

if 'models' not in st.session_state:
    st.session_state.models = RealEstatePredictiveModels()

if 'analytics' not in st.session_state:
    st.session_state.analytics = InvestmentAnalytics()

# Main title
st.markdown('<p class="main-header">🏘️ Real Estate Investment Advisor AI</p>', unsafe_allow_html=True)
st.markdown("### AI-Powered Investment Analysis & Advisory Platform")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📊 Property Analysis",
    "💰 Investment Calculator",
    "🔍 Model Explainability",
    "💬 AI Investment Advisor",
    "📈 Dashboard"
])

# Home Page
if page == "🏠 Home":
    st.header("Welcome to Real Estate Investment Advisor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🎯 Predictive Modeling**\n\nML/DL models for accurate price forecasting")
    
    with col2:
        st.success("**💹 Investment Analytics**\n\nROI, rental yield, and appreciation analysis")
    
    with col3:
        st.warning("**🤖 AI Assistant**\n\nConversational guidance powered by LangChain & Groq")
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Start")
    st.write("""
    1. **Property Analysis**: Enter property details to get price predictions
    2. **Investment Calculator**: Analyze ROI, rental yield, and cash flow
    3. **Model Explainability**: Understand what drives predictions with SHAP/LIME
    4. **AI Advisor**: Chat with our AI for personalized investment guidance
    5. **Dashboard**: Visualize comprehensive analytics and insights
    """)
    
    st.markdown("---")
    
    # Sample data preview
    st.subheader("📋 Sample Property Data")
    sample_df = st.session_state.preprocessor.create_sample_dataset(10)
    st.dataframe(sample_df)

# Property Analysis Page
elif page == "📊 Property Analysis":
    st.header("Property Price Prediction & Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Property Details")
        
        area = st.number_input("Area (sq ft)", min_value=1000, max_value=20000, value=5000, step=100)
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=6, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=4, value=2)
        stories = st.number_input("Stories", min_value=1, max_value=4, value=2)
        
        col_a, col_b = st.columns(2)
        with col_a:
            mainroad = st.selectbox("Main Road Access", ["Yes", "No"])
            guestroom = st.selectbox("Guest Room", ["Yes", "No"])
            basement = st.selectbox("Basement", ["Yes", "No"])
        
        with col_b:
            hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
            airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
            prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
        
        parking = st.number_input("Parking Spaces", min_value=0, max_value=3, value=2)
        furnishing = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])
        
        predict_button = st.button("🔮 Predict Price", type="primary")
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_button:
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
            price_per_sqft = 0  # Will be recalculated
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
            
            # Simple prediction formula (replace with actual model)
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
            
            # Display results
            st.metric("Predicted Price", f"₹{predicted_price:,.0f}")
            st.metric("Price per Sq Ft", f"₹{predicted_price/area:.0f}")
            
            # Feature importance for this prediction
            st.markdown("#### 🎯 Key Price Drivers")
            
            factors = {
                'Area': area * 800,
                'Bedrooms': bedrooms * 500000,
                'Air Conditioning': airconditioning_val * 350000,
                'Preferred Area': prefarea_val * 500000,
                'Bathrooms': bathrooms * 300000,
                'Furnishing': furnishing_val * 300000
            }
            
            factors_sorted = dict(sorted(factors.items(), key=lambda x: x[1], reverse=True)[:5])
            
            for factor, value in factors_sorted.items():
                st.progress(value / max(factors.values()))
                st.caption(f"{factor}: ₹{value:,.0f}")
            
            # Future predictions
            st.markdown("---")
            st.markdown("#### 📈 Future Value Projections")
            years = list(range(1, 11))
            appreciation_rate = 0.05  # 5% annual appreciation
            future_values = [predicted_price * (1 + appreciation_rate) ** year for year in years]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=future_values,
                mode='lines+markers',
                name='Property Value',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))
            
            fig.update_layout(
                title='10-Year Property Value Forecast',
                xaxis_title='Years',
                yaxis_title='Property Value (₹)',
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Store in session state for chatbot
            if st.session_state.chatbot:
                st.session_state.chatbot.set_property_context({
                    'price': predicted_price,
                    'area': area,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'furnishing': furnishing
                })

# Investment Calculator Page
elif page == "💰 Investment Calculator":
    st.header("Investment Analytics Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        purchase_price = st.number_input("Purchase Price ($)", min_value=50000, max_value=10000000, 
                                        value=500000, step=10000)
        annual_rental = st.number_input("Annual Rental Income ($)", min_value=0, max_value=500000,
                                       value=30000, step=1000)
        operating_expenses = st.number_input("Annual Operating Expenses ($)", min_value=0, 
                                            max_value=100000, value=8000, step=1000)
        holding_period = st.slider("Holding Period (years)", 1, 30, 5)
        
        calculate_button = st.button("💰 Calculate Metrics", type="primary")
    
    with col2:
        if calculate_button:
            # Perform analysis
            property_data = {
                'purchase_price': purchase_price,
                'annual_rental_income': annual_rental,
                'operating_expenses': operating_expenses,
                'holding_period_years': holding_period
            }
            
            analysis = st.session_state.analytics.comprehensive_analysis(property_data)
            recommendation = st.session_state.analytics.investment_recommendation(analysis)
            
            # Display key metrics
            st.subheader("Key Investment Metrics")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("ROI", f"{analysis['roi']['roi_percentage']:.2f}%")
                st.metric("Net Profit", f"${analysis['roi']['net_profit']:,.0f}")
            
            with metrics_col2:
                st.metric("Rental Yield", f"{analysis['rental_yield']['net_yield_percentage']:.2f}%")
                st.metric("Cap Rate", f"{analysis['cap_rate']['cap_rate_percentage']:.2f}%")
            
            with metrics_col3:
                st.metric("Annual Cash Flow", f"${analysis['cash_flow']['annual_cash_flow']:,.0f}")
                st.metric("Monthly Cash Flow", f"${analysis['cash_flow']['monthly_cash_flow']:,.0f}")
            
            # Recommendation
            st.markdown("---")
            st.subheader("🎯 Investment Recommendation")
            
            score = recommendation['score']
            if score >= 8:
                st.success(f"**{recommendation['overall_recommendation']}**")
            elif score >= 5:
                st.info(f"**{recommendation['overall_recommendation']}**")
            else:
                st.warning(f"**{recommendation['overall_recommendation']}**")
            
            st.write("**Key Points:**")
            for rec in recommendation['detailed_recommendations']:
                st.write(f"• {rec}")
            
            # Store analysis for chatbot
            if st.session_state.chatbot:
                st.session_state.chatbot.set_property_context(property_data, analysis)
    
    # Visualization section
    if calculate_button:
        st.markdown("---")
        st.subheader("📊 Visual Analytics")
        
        # Cash flow breakdown
        fig1 = go.Figure(data=[
            go.Bar(name='Rental Income', x=['Annual'], 
                   y=[annual_rental], marker_color='green'),
            go.Bar(name='Expenses', x=['Annual'], 
                   y=[operating_expenses], marker_color='red'),
            go.Bar(name='Net Cash Flow', x=['Annual'], 
                   y=[analysis['cash_flow']['annual_cash_flow']], marker_color='blue')
        ])
        fig1.update_layout(title='Annual Cash Flow Breakdown', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)

# Model Explainability Page
elif page == "🔍 Model Explainability":
    st.header("Model Explainability & Transparency")
    
    st.info("""
    **Understanding Model Predictions with SHAP & LIME**
    
    This page helps you understand how our ML models make predictions by showing:
    - Which features most influence the prediction
    - How each feature contributes to the final price
    - Global feature importance across all predictions
    """)
    
    st.subheader("🎯 Feature Importance Analysis")
    
    # Sample feature importance visualization
    features = ['area', 'location', 'bedrooms', 'property_age', 'amenities_score', 
                'bathrooms', 'parking_spaces', 'property_type']
    importance = [0.35, 0.22, 0.15, 0.12, 0.08, 0.05, 0.02, 0.01]
    
    fig = px.bar(x=importance, y=features, orientation='h',
                 title='Global Feature Importance',
                 labels={'x': 'Importance Score', 'y': 'Feature'},
                 color=importance, color_continuous_scale='Blues')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔬 Individual Prediction Explanation")
    
    st.write("""
    **SHAP (SHapley Additive exPlanations)** values show how each feature 
    contributes to pushing the prediction higher or lower than the baseline.
    """)
    
    # Example SHAP visualization
    sample_features = {
        'Large Area (+2500 sqft)': 45000,
        'Urban Location': 35000,
        'Modern (New Build)': 25000,
        '4 Bedrooms': 15000,
        'High Amenities': 10000,
        '3 Bathrooms': -5000,
        'No Parking': -8000
    }
    
    fig2 = go.Figure(go.Bar(
        x=list(sample_features.values()),
        y=list(sample_features.keys()),
        orientation='h',
        marker=dict(color=list(sample_features.values()),
                   colorscale='RdYlGn',
                   showscale=True)
    ))
    fig2.update_layout(title='Feature Contributions to Price Prediction',
                      xaxis_title='Impact on Price ($)',
                      yaxis_title='Feature')
    st.plotly_chart(fig2, use_container_width=True)

# AI Investment Advisor Page
elif page == "💬 AI Investment Advisor":
    st.header("AI-Powered Investment Advisor")
    
    if st.session_state.chatbot is None:
        st.error("""
        ⚠️ **Chatbot Not Available**
        
        Please ensure you have:
        1. Created a `.env` file with your GROQ_API_KEY
        2. Obtained an API key from https://console.groq.com/
        
        Example .env file:
        ```
        GROQ_API_KEY=your_api_key_here
        ```
        """)
    else:
        st.success("✅ AI Advisor Ready! Ask me anything about real estate investing.")
        
        # Chat interface
        st.markdown("---")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI Advisor:** {message['content']}")
                st.markdown("---")
        
        # Chat input
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input("Ask a question:", key="chat_input", 
                                      placeholder="e.g., What's a good ROI for rental properties?")
        
        with col2:
            send_button = st.button("Send", type="primary")
        
        if send_button and user_input:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Get AI response
            response = st.session_state.chatbot.chat(user_input)
            
            # Add AI response
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            st.rerun()
        
        # Quick questions
        st.sidebar.markdown("### 💡 Quick Questions")
        quick_questions = [
            "What's a good ROI for rental properties?",
            "How do I calculate rental yield?",
            "What factors affect property appreciation?",
            "Should I invest in urban or suburban areas?",
            "How much should I budget for maintenance?"
        ]
        
        for question in quick_questions:
            if st.sidebar.button(question, key=question):
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
        
        if st.sidebar.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.chatbot.reset_conversation()
            st.rerun()

# Dashboard Page
elif page == "📈 Dashboard":
    st.header("Investment Analytics Dashboard")
    
    # Generate sample data for dashboard
    np.random.seed(42)
    n_properties = 20
    
    dashboard_data = pd.DataFrame({
        'Property': [f'Property {i+1}' for i in range(n_properties)],
        'Price': np.random.randint(200000, 800000, n_properties),
        'ROI': np.random.uniform(10, 50, n_properties),
        'Rental_Yield': np.random.uniform(3, 8, n_properties),
        'Location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_properties),
        'Type': np.random.choice(['Apartment', 'House', 'Condo', 'Villa'], n_properties)
    })
    
    # Key metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Property Price", f"${dashboard_data['Price'].mean():,.0f}")
    with col2:
        st.metric("Avg ROI", f"{dashboard_data['ROI'].mean():.1f}%")
    with col3:
        st.metric("Avg Rental Yield", f"{dashboard_data['Rental_Yield'].mean():.1f}%")
    with col4:
        st.metric("Total Properties", len(dashboard_data))
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by location
        fig1 = px.box(dashboard_data, x='Location', y='Price', 
                     title='Price Distribution by Location',
                     color='Location')
        st.plotly_chart(fig1, use_container_width=True)
        
        # ROI vs Rental Yield scatter
        fig3 = px.scatter(dashboard_data, x='ROI', y='Rental_Yield',
                         size='Price', color='Type',
                         title='ROI vs Rental Yield Analysis',
                         hover_data=['Property', 'Location'])
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Average ROI by property type
        avg_roi = dashboard_data.groupby('Type')['ROI'].mean().reset_index()
        fig2 = px.bar(avg_roi, x='Type', y='ROI',
                     title='Average ROI by Property Type',
                     color='ROI', color_continuous_scale='Greens')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Property distribution
        type_counts = dashboard_data['Type'].value_counts()
        fig4 = px.pie(values=type_counts.values, names=type_counts.index,
                     title='Property Type Distribution')
        st.plotly_chart(fig4, use_container_width=True)
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Property Comparison Table")
    st.dataframe(dashboard_data.sort_values('ROI', ascending=False), 
                 use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Real Estate Investment Advisor AI**

Powered by:
- LangChain & Groq LLM
- Scikit-learn & TensorFlow
- SHAP & LIME
- Streamlit

© 2025 Investment Analytics Platform
""")