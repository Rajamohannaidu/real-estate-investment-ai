# tests/test_utils.py

import pytest
from utils import (
    format_currency,
    format_percentage,
    calculate_mortgage_payment,
    validate_property_data
)

class TestUtils:
    
    def test_format_currency(self):
        assert format_currency(1000) == "$1,000.00"
        assert format_currency(1000000) == "$1,000,000.00"
    
    def test_format_percentage(self):
        assert format_percentage(5.5) == "5.50%"
        assert format_percentage(10) == "10.00%"
    
    def test_calculate_mortgage_payment(self):
        payment = calculate_mortgage_payment(500000, 0.05, 30)
        assert payment > 0
        assert payment < 10000  # Sanity check
    
    def test_validate_property_data(self):
        valid_data = {
            'area': 1500,
            'bedrooms': 3,
            'bathrooms': 2,
            'year_built': 2010,
            'location': 'Urban',
            'property_type': 'House'
        }
        
        is_valid, error = validate_property_data(valid_data)
        assert is_valid is True
        assert error is None
        
        # Test invalid data
        invalid_data = valid_data.copy()
        invalid_data['area'] = -100
        is_valid, error = validate_property_data(invalid_data)
        assert is_valid is False
        assert error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])