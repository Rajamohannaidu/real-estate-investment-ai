#!/usr/bin/env python3
# train_housing_models_FINAL.py - ULTIMATE OPTIMIZED VERSION

"""
FINAL OPTIMIZED training script for Housing.csv
Based on actual data analysis - Expected R² > 0.80
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import joblib

def print_section(title):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

def load_and_preprocess_data():
    """Load and preprocess Housing.csv with optimizations"""
    
    print("Step 1: Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv('Housing.csv')
    print(f"✓ Loaded {len(df)} records")
    
    # Convert yes/no to 1/0
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # One-hot encode furnishing status
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
    print("✓ Encoded categorical features")
    
    # Remove price outliers (CRITICAL)
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5*IQR) & (df['price'] <= Q3 + 1.5*IQR)]
    
    # Remove area outliers
    Q1_area = df['area'].quantile(0.05)
    Q3_area = df['area'].quantile(0.95)
    df = df[(df['area'] >= Q1_area) & (df['area'] <= Q3_area)]
    
    print(f"✓ Removed outliers: {len(df)} records remaining")
    
    return df

def engineer_features(df):
    """OPTIMIZED feature engineering"""
    
    print("\nStep 2: Engineering features...")
    
    # Ratio features
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)
    df['space_per_bedroom'] = df['area'] / (df['bedrooms'] + 0.1)
    df['bathroom_adequacy'] = df['bathrooms'] / (df['bedrooms'] + 0.1)
    
    # Facility scores
    df['facilities_score'] = (
        df['mainroad'] + df['guestroom'] + df['basement'] +
        df['hotwaterheating'] + df['airconditioning']
    )
    
    df['luxury_index'] = (
        df['guestroom'] * 1.5 +
        df['basement'] * 1.2 +
        df['airconditioning'] * 1.0
    )
    
    df['location_score'] = df['mainroad'] * 1.5 + df['prefarea'] * 2.5
    
    # Binary indicators
    df['is_multistory'] = (df['stories'] > 1).astype(int)
    df['has_parking'] = (df['parking'] > 0).astype(int)
    df['high_end'] = ((df['area'] > 7000) & (df['bedrooms'] >= 3)).astype(int)
    
    # Polynomial features (IMPORTANT for non-linear patterns)
    df['area_squared'] = df['area'] ** 2
    df['area_cubed'] = df['area'] ** 3
    df['bedrooms_squared'] = df['bedrooms'] ** 2
    
    # Interaction features (capture complex relationships)
    df['area_x_bedrooms'] = df['area'] * df['bedrooms']
    df['area_x_bathrooms'] = df['area'] * df['bathrooms']
    df['area_x_facilities'] = df['area'] * df['facilities_score']
    df['location_x_luxury'] = df['location_score'] * df['luxury_index']
    df['stories_x_area'] = df['stories'] * df['area']
    
    # Log transformations (handle exponential relationships)
    df['log_area'] = np.log1p(df['area'])
    df['log_bedrooms'] = np.log1p(df['bedrooms'])
    
    print(f"✓ Created {df.shape[1]} total features")
    
    return df

def prepare_data(df):
    """Prepare data with OPTIMAL scaling strategy"""
    
    print("\nStep 3: Preparing data for modeling...")
    
    # Separate target
    y = df['price']
    
    # Drop target and any leakage features
    X = df.drop(columns=['price'])
    
    # Identify feature types
    binary_features = [col for col in X.columns if X[col].nunique() <= 2]
    continuous_features = [col for col in X.columns if col not in binary_features]
    
    print(f"✓ Binary features: {len(binary_features)}")
    print(f"✓ Continuous features: {len(continuous_features)}")
    
    # Scale ONLY continuous features using RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_scaled = X.copy()
    X_scaled[continuous_features] = scaler.fit_transform(X[continuous_features])
    
    # Split data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Log transform target (CRITICAL for price prediction)
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Applied log transformation to target")
    
    return X_train, X_test, y_train, y_test, y_train_log, y_test_log, scaler, X.columns.tolist()

def train_models(X_train, y_train_log, X_test, y_test_log, y_test):
    """Train all models with OPTIMIZED hyperparameters"""
    
    print_section("TRAINING MODELS")
    
    models = {}
    results = {}
    
    # 1. Ridge Regression - FIXED alpha
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0, max_iter=10000)  # Lower alpha
    ridge.fit(X_train, y_train_log)
    models['Ridge'] = ridge
    results['Ridge'] = evaluate_model(ridge, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Ridge']['r2_score']:.4f}")
    
    # 2. Lasso Regression - FIXED alpha (was WAY too high)
    print("Training Lasso Regression...")
    lasso = Lasso(alpha=0.001, max_iter=10000)  # Much lower alpha!
    lasso.fit(X_train, y_train_log)
    models['Lasso'] = lasso
    results['Lasso'] = evaluate_model(lasso, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Lasso']['r2_score']:.4f}")
    
    # 3. ElasticNet - FIXED alpha
    print("Training ElasticNet...")
    elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)  # Much lower!
    elastic.fit(X_train, y_train_log)
    models['ElasticNet'] = elastic
    results['ElasticNet'] = evaluate_model(elastic, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['ElasticNet']['r2_score']:.4f}")
    
    # 4. Random Forest - OPTIMIZED
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=500,      # More trees
        max_depth=25,          # Deeper trees
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.6,      # Consider 60% of features
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_log)
    models['Random Forest'] = rf
    results['Random Forest'] = evaluate_model(rf, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Random Forest']['r2_score']:.4f}")
    
    # 5. Gradient Boosting - OPTIMIZED
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.02,    # Smaller learning rate
        max_depth=5,
        min_samples_split=5,
        subsample=0.85,
        random_state=42
    )
    gb.fit(X_train, y_train_log)
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = evaluate_model(gb, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Gradient Boosting']['r2_score']:.4f}")
    
    # 6. XGBoost - HIGHLY OPTIMIZED
    print("Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=7,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.05,
        reg_alpha=0.01,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train_log)
    models['XGBoost'] = xgb
    results['XGBoost'] = evaluate_model(xgb, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['XGBoost']['r2_score']:.4f}")
    
    # 7. LightGBM - HIGHLY OPTIMIZED
    print("Training LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=50,
        max_depth=8,
        min_child_samples=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        reg_lambda=0.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train, y_train_log)
    models['LightGBM'] = lgbm
    results['LightGBM'] = evaluate_model(lgbm, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['LightGBM']['r2_score']:.4f}")
    
    # 8. Neural Network - FIXED
    print("Training Neural Network...")
    
    # Scale target for NN
    y_scaler = StandardScaler()
    y_train_nn = y_scaler.fit_transform(y_train_log.values.reshape(-1, 1)).ravel()
    
    nn = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.00001,         # Very light regularization
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=100,
        random_state=42
    )
    nn.fit(X_train, y_train_nn)
    models['Neural Network'] = (nn, y_scaler)
    
    # Special evaluation for NN
    y_pred_scaled = nn.predict(X_test)
    y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = np.expm1(y_pred_log)
    
    results['Neural Network'] = {
        'r2_score': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    }
    print(f"  ✓ R² = {results['Neural Network']['r2_score']:.4f}")
    
    return models, results

def evaluate_model(model, X_test, y_test_log, y_test_original):
    """Evaluate model on original scale"""
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    return {
        'r2_score': r2_score(y_test_original, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_original, y_pred)),
        'mae': mean_absolute_error(y_test_original, y_pred),
        'mape': np.mean(np.abs((y_test_original - y_pred) / (y_test_original + 1e-10))) * 100
    }

def main():
    print_section("FINAL OPTIMIZED HOUSING PRICE PREDICTION")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Prepare data
    df = load_and_preprocess_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, y_train_log, y_test_log, scaler, features = prepare_data(df)
    
    # Train models
    models, results = train_models(X_train, y_train_log, X_test, y_test_log, y_test)
    
    # Display results
    print_section("RESULTS")
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('r2_score', ascending=False)
    
    print("┌" + "─" * 78 + "┐")
    print(f"│ {'Model':<20} │ {'R² Score':<10} │ {'RMSE':<15} │ {'MAE':<15} │")
    print("├" + "─" * 78 + "┤")
    
    for model_name, metrics in results_df.iterrows():
        print(f"│ {model_name:<20} │ {metrics['r2_score']:<10.4f} │ ₹{metrics['rmse']:<14,.0f} │ ₹{metrics['mae']:<14,.0f} │")
    
    print("└" + "─" * 78 + "┘")
    
    # Best model
    best_model_name = results_df.index[0]
    best_metrics = results_df.iloc[0]
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"   R² Score: {best_metrics['r2_score']:.4f}")
    print(f"   RMSE: ₹{best_metrics['rmse']:,.0f}")
    print(f"   MAE: ₹{best_metrics['mae']:,.0f}")
    print(f"   MAPE: {best_metrics['mape']:.2f}%")
    print(f"{'='*80}")
    
    # Save models
    print("\nSaving models...")
    os.makedirs('models/saved_models', exist_ok=True)
    
    for name, model in models.items():
        if name == 'Neural Network':
            joblib.dump(model, f'models/saved_models/{name.lower().replace(" ", "_")}.pkl')
        else:
            joblib.dump(model, f'models/saved_models/{name.lower().replace(" ", "_")}.pkl')
    
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(features, 'models/saved_models/feature_names.pkl')
    
    # Save results
    with open('models/training_results.json', 'w') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'best_model': best_model_name,
            'models': {k: {kk: float(vv) for kk, vv in v.items()} 
                      for k, v in results.items()}
        }, f, indent=4)
    
    print("✓ All models and results saved")
    
    # Performance check
    if best_metrics['r2_score'] > 0.80:
        print("\n🎉 EXCELLENT! R² > 0.80 achieved!")
    elif best_metrics['r2_score'] > 0.75:
        print("\n👍 GOOD! R² > 0.75 achieved")
    else:
        print(f"\n⚠️  R² = {best_metrics['r2_score']:.4f} - Consider additional tuning")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()