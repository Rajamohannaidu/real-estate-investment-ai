# src/model_trainer.py - Train and evaluate models in real-time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow import keras
import time
import pickle
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    """Train and evaluate multiple models with real metrics"""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = None  # Added for scaling
        
    def prepare_data(self, df, target_col='price', test_size=0.2):
        """Prepare training and testing data with scaling and polynomial features"""
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # New: Add polynomial features for improvement (interactions only to avoid too many cols)
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_poly = poly.fit_transform(X)
        self.feature_names = poly.get_feature_names_out(X.columns.tolist())
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_poly, y, test_size=test_size, random_state=42
        )
        
        # New: Scale features (fit on train only)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame for compatibility (optional, but helps with naming)
        self.X_train = pd.DataFrame(self.X_train, columns=self.feature_names)
        self.X_test = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_all_models(self, quick_mode=False):
        """Train all models and collect metrics (with optional tuning)"""
        
        print("🚀 Starting model training...")
        
        # Define models with improved hyperparameters
        models_to_train = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100 if quick_mode else 200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100 if quick_mode else 200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100 if quick_mode else 150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=100 if quick_mode else 200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        # New: Optional hyperparameter tuning (for non-quick mode, tune top models)
        if not quick_mode:
            print("  Performing hyperparameter tuning for select models...")
            # Example for Random Forest
            rf_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10]
            }
            rf_grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
            rf_grid.fit(self.X_train, self.y_train)
            models_to_train['Random Forest'] = rf_grid.best_estimator_
            print(f"    ✓ Best Random Forest params: {rf_grid.best_params_}")
            
            # Similarly for XGBoost (add more as needed)
            xgb_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1]
            }
            xgb_grid = GridSearchCV(XGBRegressor(random_state=42, n_jobs=-1), xgb_param_grid, cv=5, scoring='r2', n_jobs=-1)
            xgb_grid.fit(self.X_train, self.y_train)
            models_to_train['XGBoost'] = xgb_grid.best_estimator_
            print(f"    ✓ Best XGBoost params: {xgb_grid.best_params_}")
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            
            start_time = time.time()
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Calculate metrics
            metrics = {
                'model': model,
                'r2_train': r2_score(self.y_train, y_pred_train),
                'r2_test': r2_score(self.y_test, y_pred_test),
                'rmse_train': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'rmse_test': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'mae_train': mean_absolute_error(self.y_train, y_pred_train),
                'mae_test': mean_absolute_error(self.y_test, y_pred_test),
                'training_time': training_time
            }
            
            # Cross-validation score (if not too slow)
            if not quick_mode or name in ['Ridge', 'Linear Regression']:
                try:
                    cv_scores = cross_val_score(
                        model, self.X_train, self.y_train, 
                        cv=5, scoring='r2', n_jobs=-1
                    )
                    metrics['cv_r2_mean'] = cv_scores.mean()
                    metrics['cv_r2_std'] = cv_scores.std()
                except:
                    metrics['cv_r2_mean'] = None
                    metrics['cv_r2_std'] = None
            
            self.models[name] = model
            self.metrics[name] = metrics
            
            print(f"    ✓ {name}: R² = {metrics['r2_test']:.4f}, RMSE = ₹{metrics['rmse_test']:,.0f}")
        
        # Train Neural Network (if not quick mode)
        if not quick_mode:
            print("  Training Neural Network...")
            nn_metrics = self._train_neural_network()
            if nn_metrics:
                self.metrics['Neural Network'] = nn_metrics
                print(f"    ✓ Neural Network: R² = {nn_metrics['r2_test']:.4f}, RMSE = ₹{nn_metrics['rmse_test']:,.0f}")
        
        # Find best model
        self._find_best_model()
        
        print(f"\n✅ Training complete! Best model: {self.best_model_name}")
        
        return self.metrics
    
    def _train_neural_network(self):
        """Train a neural network model"""
        try:
            # Build model (added more layers for improvement)
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(16, activation='relu'),  # New layer
                keras.layers.Dense(1)
            ])
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            start_time = time.time()
            
            # Train with early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                self.X_train, self.y_train,
                validation_split=0.2,
                epochs=200,  # Increased epochs
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred_train = model.predict(self.X_train, verbose=0).flatten()
            y_pred_test = model.predict(self.X_test, verbose=0).flatten()
            
            self.models['Neural Network'] = model
            
            return {
                'model': model,
                'r2_train': r2_score(self.y_train, y_pred_train),
                'r2_test': r2_score(self.y_test, y_pred_test),
                'rmse_train': np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                'rmse_test': np.sqrt(mean_squared_error(self.y_test, y_pred_test)),
                'mae_train': mean_absolute_error(self.y_train, y_pred_train),
                'mae_test': mean_absolute_error(self.y_test, y_pred_test),
                'training_time': training_time,
                'cv_r2_mean': None,
                'cv_r2_std': None
            }
            
        except Exception as e:
            print(f"    ✗ Neural Network training failed: {e}")
            return None
    
    def _find_best_model(self):
        """Find the best performing model based on test R²"""
        best_r2 = -float('inf')
        
        for name, metrics in self.metrics.items():
            if metrics['r2_test'] > best_r2:
                best_r2 = metrics['r2_test']
                self.best_model_name = name
                self.best_model = metrics['model']
    
    def get_metrics_dataframe(self):
        """Convert metrics to a pandas DataFrame"""
        
        metrics_list = []
        
        for name, metrics in self.metrics.items():
            metrics_list.append({
                'Model': name,
                'R² Score (Test)': metrics['r2_test'],
                'R² Score (Train)': metrics['r2_train'],
                'RMSE (Test)': metrics['rmse_test'],
                'MAE (Test)': metrics['mae_test'],
                'Training Time (s)': metrics['training_time'],
                'CV R² Mean': metrics.get('cv_r2_mean', None),
                'CV R² Std': metrics.get('cv_r2_std', None),
                'Overfit Score': metrics['r2_train'] - metrics['r2_test']
            })
        
        df = pd.DataFrame(metrics_list).sort_values('R² Score (Test)', ascending=False)
        return df
    
    def get_feature_importance(self, model_name=None):
        """Get feature importance from the specified model"""
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # For models without built-in importance, use permutation
            from sklearn.inspection import permutation_importance
            
            perm_importance = permutation_importance(
                model, self.X_test, self.y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )
            importance = perm_importance.importances_mean
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Normalize
        importance_df['importance'] = importance_df['importance'] / importance_df['importance'].sum()
        
        return importance_df
    
    def predict(self, X, model_name=None, scale=True):
        """Make predictions using specified model (optional scaling for input)"""
        
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Scale the input if requested (e.g., for raw app inputs)
        if scale:
            X = self.scaler.transform(X)
        
        model = self.models[model_name]
        
        if model_name == 'Neural Network':
            return model.predict(X, verbose=0).flatten()
        else:
            return model.predict(X)
    
    def save_models(self, directory='models'):
        """Save all trained models"""
        
        os.makedirs(directory, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'Neural Network':
                model.save(f'{directory}/neural_network.keras')  # Updated to .keras to avoid warning
            else:
                with open(f'{directory}/{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        # Save metrics
        metrics_df = self.get_metrics_dataframe()
        metrics_df.to_csv(f'{directory}/model_metrics.csv', index=False)
        
        print(f"✅ Models saved to {directory}/")
    
    def load_models(self, directory='models'):
        """Load saved models"""
        
        if not os.path.exists(directory):
            return False
        
        # Load metrics
        metrics_path = f'{directory}/model_metrics.csv'
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            print(f"✅ Loaded model metrics from {metrics_path}")
        
        return True
    
    def compare_models(self):
        """Generate comparison visualizations"""
        
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        df = self.get_metrics_dataframe()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Score Comparison', 'RMSE Comparison', 
                          'Training Time', 'Train vs Test R²'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # R² Score
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['R² Score (Test)'], 
                   name='R² Score', marker_color='#667eea'),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['RMSE (Test)'], 
                   name='RMSE', marker_color='#fc8181'),
            row=1, col=2
        )
        
        # Training Time
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Training Time (s)'], 
                   name='Time', marker_color='#8ec5fc'),
            row=2, col=1
        )
        
        # Train vs Test
        fig.add_trace(
            go.Scatter(x=df['R² Score (Train)'], y=df['R² Score (Test)'],
                      mode='markers+text', text=df['Model'],
                      textposition='top center', marker=dict(size=10, color='#667eea')),
            row=2, col=2
        )
        
        # Add diagonal line for perfect fit
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                      line=dict(dash='dash', color='gray'),
                      showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Model Performance Comparison")
        
        return fig