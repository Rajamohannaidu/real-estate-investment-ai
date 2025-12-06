# app/components/prediction_view.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_prediction_view(preprocessor, models):
    """Render the property prediction interface"""
    
    st.header("🏠 Property Price Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Property Details")
        
        with st.form("property_form"):
            area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1500)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
            year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010)
            
            location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
            property_type = st.selectbox("Property Type", ["Apartment", "House", "Condo", "Villa"])
            
            parking_spaces = st.number_input("Parking Spaces", min_value=0, max_value=5, value=2)
            amenities_score = st.slider("Amenities Score (1-10)", 1, 10, 7)
            
            submitted = st.form_submit_button("🔮 Predict Price", type="primary")
    
    with col2:
        if submitted:
            render_prediction_results(
                area, bedrooms, bathrooms, year_built, location, 
                property_type, parking_spaces, amenities_score, 
                preprocessor, models
            )

def render_prediction_results(area, bedrooms, bathrooms, year_built, location, 
                              property_type, parking_spaces, amenities_score,
                              preprocessor, models):
    """Display prediction results with visualizations"""
    
    st.subheader("Prediction Results")
    
    # Simple prediction (replace with actual model prediction)
    base_price = (
        area * 150 + 
        bedrooms * 50000 + 
        bathrooms * 30000 + 
        (2025 - year_built) * -2000 +
        parking_spaces * 20000 +
        amenities_score * 10000
    )
    
    location_multiplier = {'Urban': 1.5, 'Suburban': 1.2, 'Rural': 0.8}[location]
    predicted_price = base_price * location_multiplier
    
    # Display main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Price", f"${predicted_price:,.0f}")
    with col2:
        st.metric("Price per Sq Ft", f"${predicted_price/area:.2f}")
    with col3:
        st.metric("Property Age", f"{2025 - year_built} years")
    
    # Future value projections
    st.markdown("#### 📈 Future Value Projections")
    
    years = list(range(1, 11))
    appreciation_rate = 0.04
    future_values = [predicted_price * (1 + appreciation_rate) ** year for year in years]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=future_values,
        mode='lines+markers',
        name='Property Value',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='10-Year Property Value Forecast',
        xaxis_title='Years',
        yaxis_title='Property Value ($)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    with st.expander("📊 View Detailed Breakdown"):
        breakdown_data = {
            'Component': ['Base Area Value', 'Bedrooms', 'Bathrooms', 'Location Premium', 
                         'Parking', 'Amenities', 'Age Adjustment'],
            'Value': [area * 150, bedrooms * 50000, bathrooms * 30000, 
                     base_price * (location_multiplier - 1),
                     parking_spaces * 20000, amenities_score * 10000,
                     (2025 - year_built) * -2000]
        }
        breakdown_df = pd.DataFrame(breakdown_data)
        
        fig2 = px.bar(breakdown_df, x='Component', y='Value', 
                     title='Price Component Breakdown',
                     color='Value', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig2, use_container_width=True)


