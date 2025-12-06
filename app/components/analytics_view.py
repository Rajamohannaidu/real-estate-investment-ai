# app/components/analytics_view.py

def render_analytics_view(analytics):
    """Render investment analytics interface"""
    
    st.header("💰 Investment Analytics Calculator")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Investment Parameters")
        
        with st.form("analytics_form"):
            purchase_price = st.number_input(
                "Purchase Price ($)", 
                min_value=50000, 
                max_value=10000000, 
                value=500000, 
                step=10000
            )
            
            annual_rental = st.number_input(
                "Annual Rental Income ($)", 
                min_value=0, 
                max_value=500000,
                value=30000, 
                step=1000
            )
            
            operating_expenses = st.number_input(
                "Annual Operating Expenses ($)", 
                min_value=0, 
                max_value=100000, 
                value=8000, 
                step=1000
            )
            
            holding_period = st.slider("Holding Period (years)", 1, 30, 5)
            
            mortgage_payment = st.number_input(
                "Annual Mortgage Payment ($)", 
                min_value=0, 
                value=0, 
                step=1000
            )
            
            calculated = st.form_submit_button("💹 Calculate Metrics", type="primary")
    
    with col2:
        if calculated:
            render_analytics_results(
                purchase_price, annual_rental, operating_expenses, 
                holding_period, mortgage_payment, analytics
            )

def render_analytics_results(purchase_price, annual_rental, operating_expenses,
                             holding_period, mortgage_payment, analytics):
    """Display investment analytics results"""
    
    st.subheader("Investment Analysis Results")
    
    property_data = {
        'purchase_price': purchase_price,
        'annual_rental_income': annual_rental,
        'operating_expenses': operating_expenses,
        'holding_period_years': holding_period
    }
    
    analysis = analytics.comprehensive_analysis(property_data)
    recommendation = analytics.investment_recommendation(analysis)
    
    # Key metrics
    st.markdown("#### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROI", f"{analysis['roi']['roi_percentage']:.2f}%")
    with col2:
        st.metric("Rental Yield", f"{analysis['rental_yield']['net_yield_percentage']:.2f}%")
    with col3:
        st.metric("Cap Rate", f"{analysis['cap_rate']['cap_rate_percentage']:.2f}%")
    with col4:
        st.metric("Cash Flow", f"${analysis['cash_flow']['annual_cash_flow']:,.0f}")
    
    # Recommendation
    st.markdown("---")
    st.markdown("#### 🎯 Investment Recommendation")
    
    score = recommendation['score']
    if score >= 8:
        st.success(f"**{recommendation['overall_recommendation']}**")
    elif score >= 5:
        st.info(f"**{recommendation['overall_recommendation']}**")
    else:
        st.warning(f"**{recommendation['overall_recommendation']}**")
    
    # Progress bar for score
    st.progress(score / 10)
    st.caption(f"Investment Score: {score}/10")
    
    # Detailed recommendations
    with st.expander("📋 Detailed Analysis"):
        for rec in recommendation['detailed_recommendations']:
            st.write(f"✓ {rec}")
    
    # Visualizations
    render_analytics_charts(analysis)

def render_analytics_charts(analysis):
    """Render investment analytics charts"""
    
    st.markdown("---")
    st.markdown("#### 📈 Visual Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cash flow breakdown
        income = analysis['cash_flow']['effective_rental_income']
        expenses = analysis['cash_flow']['total_annual_expenses']
        net_flow = analysis['cash_flow']['annual_cash_flow']
        
        fig1 = go.Figure(data=[
            go.Bar(name='Income', x=['Annual'], y=[income], marker_color='green'),
            go.Bar(name='Expenses', x=['Annual'], y=[expenses], marker_color='red'),
            go.Bar(name='Net Cash Flow', x=['Annual'], y=[net_flow], marker_color='blue')
        ])
        fig1.update_layout(title='Annual Cash Flow Breakdown', barmode='group')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # ROI over time
        years = list(range(1, 6))
        roi_values = [analysis['roi']['roi_percentage'] * (year / 5) for year in years]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=years, y=roi_values,
            mode='lines+markers',
            name='Cumulative ROI',
            line=dict(color='green', width=3)
        ))
        fig2.update_layout(
            title='ROI Growth Over Time',
            xaxis_title='Years',
            yaxis_title='ROI (%)'
        )
        st.plotly_chart(fig2, use_container_width=True)


