# src/predictive_models.py - IMPROVED VERSION

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

class RealEstatePredictiveModels:
    """Improved ML models for real estate price prediction with optimized hyperparameters"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = self.models  # Alias for compatibility
        self.model_metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.use_log_target = True
        self.nn_target_scaler = None  # Separate scaler for Neural Network
        
    def train_all_models(self, X_train, y_train, X_test=None, y_test=None, use_log=True):
        """
        Train all models with improved hyperparameters
        
        Parameters:
        - X_train, y_train: Training data
        - X_test, y_test: Test data (can be original or log-transformed)
        - use_log: If True, assumes y_train/y_test are already log-transformed
        """
        results = {}
        self.use_log_target = use_log
        
        print("\n" + "="*80)
        print("TRAINING MODELS WITH IMPROVED HYPERPARAMETERS")
        print("="*80 + "\n")
        
        # 1. Ridge Regression (L2 regularization)
        print("Training Ridge Regression...")
        ridge = Ridge(alpha=10.0, max_iter=5000)
        ridge.fit(X_train, y_train)
        self.models['ridge'] = ridge
        
        if X_test is not None and y_test is not None:
            results['ridge'] = self.evaluate_model(ridge, X_test, y_test, use_log)
            self.model_metrics['Ridge'] = results['ridge']
            print(f"  ✓ R² = {results['ridge']['r2_score']:.4f}")
        
        # 2. Lasso Regression (L1 regularization)
        print("Training Lasso Regression...")
        lasso = Lasso(alpha=100.0, max_iter=5000)
        lasso.fit(X_train, y_train)
        self.models['lasso'] = lasso
        
        if X_test is not None and y_test is not None:
            results['lasso'] = self.evaluate_model(lasso, X_test, y_test, use_log)
            self.model_metrics['Lasso'] = results['lasso']
            print(f"  ✓ R² = {results['lasso']['r2_score']:.4f}")
        
        # 3. ElasticNet (L1 + L2 regularization)
        print("Training ElasticNet...")
        elastic = ElasticNet(alpha=10.0, l1_ratio=0.5, max_iter=5000)
        elastic.fit(X_train, y_train)
        self.models['elasticnet'] = elastic
        
        if X_test is not None and y_test is not None:
            results['elasticnet'] = self.evaluate_model(elastic, X_test, y_test, use_log)
            self.model_metrics['ElasticNet'] = results['elasticnet']
            print(f"  ✓ R² = {results['elasticnet']['r2_score']:.4f}")
        
        # 4. Random Forest (tuned hyperparameters)
        print("Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        if X_test is not None and y_test is not None:
            results['random_forest'] = self.evaluate_model(rf, X_test, y_test, use_log)
            self.model_metrics['Random Forest'] = results['random_forest']
            print(f"  ✓ R² = {results['random_forest']['r2_score']:.4f}")
        
        # 5. Gradient Boosting (tuned hyperparameters)
        print("Training Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        
        if X_test is not None and y_test is not None:
            results['gradient_boosting'] = self.evaluate_model(gb, X_test, y_test, use_log)
            self.model_metrics['Gradient Boosting'] = results['gradient_boosting']
            print(f"  ✓ R² = {results['gradient_boosting']['r2_score']:.4f}")
        
        # 6. XGBoost (tuned hyperparameters)
        print("Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb.fit(X_train, y_train)
        self.models['xgboost'] = xgb
        
        if X_test is not None and y_test is not None:
            results['xgboost'] = self.evaluate_model(xgb, X_test, y_test, use_log)
            self.model_metrics['XGBoost'] = results['xgboost']
            print(f"  ✓ R² = {results['xgboost']['r2_score']:.4f}")
        
        # 7. LightGBM (tuned hyperparameters)
        print("Training LightGBM...")
        lgbm = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)
        self.models['lightgbm'] = lgbm
        
        if X_test is not None and y_test is not None:
            results['lightgbm'] = self.evaluate_model(lgbm, X_test, y_test, use_log)
            self.model_metrics['LightGBM'] = results['lightgbm']
            print(f"  ✓ R² = {results['lightgbm']['r2_score']:.4f}")
        
        # 8. Neural Network (FIXED with proper scaling)
        print("Training Neural Network...")
        
        # Neural networks need scaled targets for stability
        self.nn_target_scaler = StandardScaler()
        
        if use_log:
            # If input is log-transformed, use it directly but scale it
            y_train_nn = self.nn_target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        else:
            # Scale original target
            y_train_nn = self.nn_target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        
        nn = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            random_state=42,
            verbose=False
        )
        nn.fit(X_train, y_train_nn)
        self.models['neural_network'] = nn
        
        if X_test is not None and y_test is not None:
            # Special evaluation for NN
            results['neural_network'] = self.evaluate_neural_network(nn, X_test, y_test, use_log)
            self.model_metrics['Neural Network'] = results['neural_network']
            print(f"  ✓ R² = {results['neural_network']['r2_score']:.4f}")
        
        # Find best model
        if results:
            best_r2 = max(results.items(), key=lambda x: x[1]['r2_score'])
            self.best_model_name = best_r2[0]
            self.best_model = self.models[best_r2[0]]
            
            # Convert to display name
            display_name = self.get_display_name(self.best_model_name)
            
            print(f"\n{'='*80}")
            print(f"🏆 BEST MODEL: {display_name}")
            print(f"   R² Score: {best_r2[1]['r2_score']:.4f}")
            print(f"   RMSE: ₹{best_r2[1]['rmse']:,.0f}")
            print(f"   MAE: ₹{best_r2[1]['mae']:,.0f}")
            print(f"{'='*80}\n")
        
        return results
    
    def evaluate_neural_network(self, model, X_test, y_test, use_log):
        """Special evaluation for neural network with proper scaling"""
        # Get scaled predictions
        y_pred_scaled = model.predict(X_test)
        
        # Inverse transform to get predictions in log scale (or original scale)
        y_pred = self.nn_target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # If using log target, transform predictions back to original scale for metrics
        if use_log:
            y_pred_original = np.expm1(y_pred)
            y_test_original = np.expm1(y_test)
        else:
            y_pred_original = y_pred
            y_test_original = y_test
        
        # Calculate metrics on original scale
        return self._calculate_metrics(y_test_original, y_pred_original)
    
    def evaluate_model(self, model, X_test, y_test, use_log=True):
        """Evaluate model performance"""
        # Get predictions (in log scale if use_log=True)
        y_pred = model.predict(X_test)
        
        # Transform back to original scale if using log
        if use_log:
            y_pred_original = np.expm1(y_pred)
            y_test_original = np.expm1(y_test)
        else:
            y_pred_original = y_pred
            y_test_original = y_test
        
        return self._calculate_metrics(y_test_original, y_pred_original)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate all metrics"""
        # Flatten if needed
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        # Convert to numpy array if pandas Series
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Safe MAPE calculation
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'r2': r2,  # Alias for compatibility
            'mape': mape
        }
    
    def predict(self, X, model_name=None):
        """Make predictions using specified or best model"""
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name, self.best_model)
        
        predictions = model.predict(X)
        
        # Special handling for neural network
        if model_name == 'neural_network' and self.nn_target_scaler is not None:
            # Inverse scale
            predictions = self.nn_target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).ravel()
        
        # Transform back from log scale if applicable
        if self.use_log_target:
            predictions = np.expm1(predictions)
        
        # Flatten if needed
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def get_display_name(self, internal_name):
        """Convert internal model name to display name"""
        name_map = {
            'ridge': 'Ridge',
            'lasso': 'Lasso',
            'elasticnet': 'ElasticNet',
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'xgboost': 'XGBoost',
            'lightgbm': 'LightGBM',
            'neural_network': 'Neural Network'
        }
        return name_map.get(internal_name, internal_name)
    
    def get_best_model(self):
        """Get the name of the best performing model (formatted for display)"""
        if self.best_model_name is None:
            return None
        return self.get_display_name(self.best_model_name)
    
    def save_models(self, directory='models/saved_models/'):
        """Save all trained models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f'{directory}{name}.pkl'
            joblib.dump(model, model_path)
        
        # Save neural network scaler separately
        if self.nn_target_scaler is not None:
            joblib.dump(self.nn_target_scaler, f'{directory}nn_target_scaler.pkl')
        
        print(f"✓ Saved {len(self.models)} models to {directory}")
    
    def load_models(self, directory='models/saved_models/'):
        """Load pre-trained models"""
        import os
        
        if not os.path.exists(directory):
            print(f"❌ Directory {directory} not found")
            return
        
        for filename in os.listdir(directory):
            if filename.endswith('.pkl') and filename != 'nn_target_scaler.pkl':
                name = filename.replace('.pkl', '')
                self.models[name] = joblib.load(f'{directory}{filename}')
        
        # Load neural network scaler if exists
        nn_scaler_path = f'{directory}nn_target_scaler.pkl'
        if os.path.exists(nn_scaler_path):
            self.nn_target_scaler = joblib.load(nn_scaler_path)
        
        print(f"✓ Loaded {len(self.models)} models from {directory}")
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models (compatibility method)"""
        if self.model_metrics:
            return self.model_metrics
        
        for name, model in self.models.items():
            display_name = self.get_display_name(name)
            
            if name == 'neural_network':
                metrics = self.evaluate_neural_network(model, X_test, y_test, self.use_log_target)
            else:
                metrics = self.evaluate_model(model, X_test, y_test, self.use_log_target)
            
            self.model_metrics[display_name] = metrics
        
        return self.model_metrics
    
    def get_feature_importance(self, model_name=None, feature_names=None):
        """Get feature importance for tree-based models"""
        if model_name is None:
            model_name = self.best_model_name
        
        model = self.models.get(model_name)
        
        if model is None:
            return None
        
        # Check if model has feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is not None:
                return pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            else:
                return importances
        
        return None


# Testing
if __name__ == "__main__":
    print("RealEstatePredictiveModels - Improved Version")
    print("Ready for training with optimized hyperparameters")
    print("\nExpected improvements:")
    print("  - Ridge/Lasso/ElasticNet for baseline")
    print("  - Random Forest: R² > 0.80")
    print("  - Gradient Boosting: R² > 0.82")
    print("  - XGBoost: R² > 0.83")
    print("  - LightGBM: R² > 0.82")
    print("  - Neural Network: R² > 0.75 (FIXED)")