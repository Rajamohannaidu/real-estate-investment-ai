# app/components/explainability_view.py

def render_explainability_view():
    """Render model explainability interface"""
    
    st.header("🔍 Model Explainability & Transparency")
    
    st.info("""
    **Understanding AI Predictions with SHAP & LIME**
    
    This section helps you understand:
    - Which features most influence property prices
    - How each feature contributes to specific predictions
    - Global patterns vs individual cases
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🔬 Individual Explanations", "📚 Learn More"])
    
    with tab1:
        render_feature_importance()
    
    with tab2:
        render_individual_explanation()
    
    with tab3:
        render_explainability_guide()

def render_feature_importance():
    """Display global feature importance"""
    
    st.subheader("Global Feature Importance")
    
    features = ['Area (sq ft)', 'Location', 'Bedrooms', 'Property Age', 
                'Amenities Score', 'Bathrooms', 'Parking Spaces', 'Property Type']
    importance = [0.35, 0.22, 0.15, 0.12, 0.08, 0.05, 0.02, 0.01]
    
    fig = px.bar(
        x=importance, 
        y=features, 
        orientation='h',
        title='Features Ranked by Impact on Price Predictions',
        labels={'x': 'Importance Score', 'y': 'Feature'},
        color=importance, 
        color_continuous_scale='Blues'
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    - **Area** has the highest impact - larger properties cost more
    - **Location** is crucial - urban properties command premium prices
    - **Bedrooms** strongly influence family home values
    - **Property Age** affects perceived value and condition
    """)

def render_individual_explanation():
    """Display individual prediction explanation"""
    
    st.subheader("Individual Prediction Breakdown")
    
    st.write("See how each feature contributes to a specific property's predicted price:")
    
    sample_features = {
        'Large Area (+2500 sqft)': 45000,
        'Prime Urban Location': 35000,
        'Modern Construction (2020)': 25000,
        '4 Bedrooms': 15000,
        'Premium Amenities': 10000,
        'Limited Parking': -8000,
        'High Property Tax Area': -5000
    }
    
    fig = go.Figure(go.Bar(
        x=list(sample_features.values()),
        y=list(sample_features.keys()),
        orientation='h',
        marker=dict(
            color=list(sample_features.values()),
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Impact ($)")
        )
    ))
    
    fig.update_layout(
        title='How Features Push Price Up or Down',
        xaxis_title='Price Impact ($)',
        yaxis_title='Feature',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **Predicted Price: $575,000**
    
    Base Price: $450,000
    Total Feature Impact: +$117,000
    """)

def render_explainability_guide():
    """Display explainability methodology guide"""
    
    st.subheader("Understanding Model Explanations")
    
    st.markdown("""
    ### 🎯 SHAP (SHapley Additive exPlanations)
    
    SHAP values show how much each feature contributes to the prediction:
    - **Positive SHAP value**: Feature increases the predicted price
    - **Negative SHAP value**: Feature decreases the predicted price
    - **Magnitude**: Shows strength of the feature's impact
    
    ### 🔬 LIME (Local Interpretable Model-agnostic Explanations)
    
    LIME explains individual predictions by:
    - Creating simplified models around specific predictions
    - Highlighting which features matter most for that case
    - Providing local interpretability
    
    ### ✅ Benefits
    
    - **Transparency**: See exactly why the model predicts certain prices
    - **Trust**: Verify predictions align with real-world knowledge
    - **Insights**: Learn what drives property values
    - **Debugging**: Identify when models might be biased
    """)


