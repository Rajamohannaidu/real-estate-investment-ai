# tests/test_models.py

import pytest
import numpy as np
from src.predictive_models import RealEstatePredictiveModels

class TestPredictiveModels:
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = X.sum(axis=1) * 100000 + 300000
        return X, y
    
    def test_neural_network_creation(self):
        models = RealEstatePredictiveModels()
        nn = models.build_neural_network(input_dim=5)
        assert nn is not None
        assert len(nn.layers) > 0
    
    def test_prediction(self, sample_data):
        models = RealEstatePredictiveModels()
        X, y = sample_data
        
        # Train a simple model
        from sklearn.ensemble import RandomForestRegressor
        models.models['rf'] = RandomForestRegressor(n_estimators=10, random_state=42)
        models.models['rf'].fit(X[:80], y[:80])
        models.best_model = models.models['rf']
        
        # Test prediction
        predictions = models.predict(X[80:])
        assert len(predictions) == 20
        assert all(predictions > 0)