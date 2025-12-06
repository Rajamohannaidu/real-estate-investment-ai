# tests/test_analytics.py

import pytest
from src.investment_analytics import InvestmentAnalytics

class TestInvestmentAnalytics:
    
    @pytest.fixture
    def analytics(self):
        return InvestmentAnalytics()
    
    def test_calculate_roi(self, analytics):
        result = analytics.calculate_roi(
            purchase_price=500000,
            annual_rental_income=30000,
            operating_expenses=8000,
            holding_period_years=5
        )
        
        assert 'roi_percentage' in result
        assert 'net_profit' in result
        assert result['roi_percentage'] > 0
    
    def test_calculate_rental_yield(self, analytics):
        result = analytics.calculate_rental_yield(
            purchase_price=500000,
            annual_rental_income=30000
        )
        
        assert 'gross_yield_percentage' in result
        assert 'net_yield_percentage' in result
        assert result['gross_yield_percentage'] == 6.0
    
    def test_calculate_cap_rate(self, analytics):
        result = analytics.calculate_cap_rate(
            purchase_price=500000,
            annual_rental_income=30000,
            operating_expenses=10000
        )
        
        assert 'cap_rate_percentage' in result
        assert result['cap_rate_percentage'] == 4.0
    
    def test_comprehensive_analysis(self, analytics):
        property_data = {
            'purchase_price': 500000,
            'annual_rental_income': 30000,
            'operating_expenses': 8000,
            'holding_period_years': 5
        }
        
        result = analytics.comprehensive_analysis(property_data)
        
        assert 'roi' in result
        assert 'rental_yield' in result
        assert 'appreciation' in result
        assert 'cash_flow' in result
    
    def test_investment_recommendation(self, analytics):
        analysis = {
            'roi': {'roi_percentage': 50},
            'rental_yield': {'net_yield_percentage': 7},
            'cap_rate': {'cap_rate_percentage': 8},
            'cash_flow': {'annual_cash_flow': 15000}
        }
        
        result = analytics.investment_recommendation(analysis)
        
        assert 'score' in result
        assert 'overall_recommendation' in result
        assert result['score'] >= 0 and result['score'] <= 10