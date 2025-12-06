# tests/test_preprocessing.py

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import RealEstateDataPreprocessor

class TestDataPreprocessing:
    
    @pytest.fixture
    def preprocessor(self):
        return RealEstateDataPreprocessor()
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'area': [1000, 1500, 2000],
            'bedrooms': [2, 3, 4],
            'bathrooms': [1, 2, 3],
            'year_built': [2000, 2010, 2020],
            'location': ['Urban', 'Suburban', 'Rural'],
            'property_type': ['Apartment', 'House', 'Villa'],
            'parking_spaces': [1, 2, 3],
            'amenities_score': [5, 7, 9],
            'price': [300000, 450000, 600000]
        })
    
    def test_create_sample_dataset(self, preprocessor):
        df = preprocessor.create_sample_dataset(100)
        assert len(df) == 100
        assert 'price' in df.columns
        assert df['price'].min() >= 100000
    
    def test_clean_data(self, preprocessor, sample_data):
        cleaned = preprocessor.clean_data(sample_data)
        assert len(cleaned) == len(sample_data)
        assert cleaned.isnull().sum().sum() == 0
    
    def test_feature_engineering(self, preprocessor, sample_data):
        engineered = preprocessor.feature_engineering(sample_data)
        assert 'price_per_sqft' in engineered.columns
        assert 'property_age' in engineered.columns
    
    def test_encode_categorical(self, preprocessor, sample_data):
        encoded = preprocessor.encode_categorical(sample_data, ['location', 'property_type'])
        assert encoded['location'].dtype in [np.int64, np.int32]
        assert encoded['property_type'].dtype in [np.int64, np.int32]
