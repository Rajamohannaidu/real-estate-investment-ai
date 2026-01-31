#!/usr/bin/env python3
# train_housing_models_ULTIMATE.py - DATA-DRIVEN OPTIMIZATION

"""
ULTIMATE VERSION - Based on actual Housing.csv analysis
Expected R² > 0.85 with XGBoost/LightGBM
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import joblib

def print_section(title):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

def load_and_preprocess():
    """Load with MINIMAL outlier removal (only 2.8% are real outliers)"""
    
    print("Step 1: Loading Housing.csv...")
    df = pd.read_csv('Housing.csv')
    print(f"✓ Loaded {len(df)} records")
    
    # Convert binary features
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # One-hot encode furnishing (keep all 3 categories - important signal!)
    furnish_dummies = pd.get_dummies(df['furnishingstatus'], prefix='furnish', drop_first=False)
    df = pd.concat([df.drop('furnishingstatus', axis=1), furnish_dummies], axis=1)
    
    print("✓ Encoded categorical features")
    
    # CONSERVATIVE outlier removal (only extreme outliers)
    # Data shows only 2.8% real outliers - don't be too aggressive!
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Only remove extreme outliers (3× IQR instead of 1.5×)
    lower = Q1 - 3 * IQR
    upper = Q3 + 3 * IQR
    df = df[(df['price'] >= lower) & (df['price'] <= upper)]
    
    # Keep more area data (don't remove too much)
    df = df[(df['area'] >= df['area'].quantile(0.02)) & 
            (df['area'] <= df['area'].quantile(0.98))]
    
    print(f"✓ Conservative outlier removal: {len(df)} records remaining")
    
    return df

def engineer_features_ultimate(df):
    """ULTIMATE feature engineering based on data analysis"""
    
    print("\nStep 2: Engineering features (data-driven)...")
    
    # Data shows area has 0.536 correlation with price - make it stronger!
    # Create multiple area transformations
    df['area_sqrt'] = np.sqrt(df['area'])
    df['area_squared'] = df['area'] ** 2
    df['area_cubed'] = df['area'] ** 3
    df['area_log'] = np.log1p(df['area'])
    df['area_log_squared'] = df['area_log'] ** 2
    
    # Bathrooms have 0.518 correlation - second most important!
    df['bathrooms_squared'] = df['bathrooms'] ** 2
    df['bathrooms_cubed'] = df['bathrooms'] ** 3
    
    # Stories have 0.421 correlation - underutilized!
    df['stories_squared'] = df['stories'] ** 2
    df['stories_log'] = np.log1p(df['stories'])
    
    # Bedrooms (0.366 correlation)
    df['bedrooms_squared'] = df['bedrooms'] ** 2
    df['bedrooms_log'] = np.log1p(df['bedrooms'])
    
    # CRITICAL: Interaction features (capture multiplicative effects)
    df['area_x_bathrooms'] = df['area'] * df['bathrooms']  # Area + quality
    df['area_x_stories'] = df['area'] * df['stories']      # Area + height
    df['area_x_bedrooms'] = df['area'] * df['bedrooms']    # Area + capacity
    df['area_x_parking'] = df['area'] * df['parking']      # Area + parking
    
    # Bathroom quality indicators
    df['bath_per_bedroom'] = df['bathrooms'] / (df['bedrooms'] + 0.1)
    df['bath_per_sqft'] = df['bathrooms'] / (df['area'] / 1000)
    
    # Space quality
    df['sqft_per_bedroom'] = df['area'] / (df['bedrooms'] + 0.1)
    df['sqft_per_bathroom'] = df['area'] / (df['bathrooms'] + 0.1)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['sqft_per_room'] = df['area'] / (df['total_rooms'] + 0.1)
    
    # Facility aggregations
    df['total_facilities'] = (
        df['mainroad'] + df['guestroom'] + df['basement'] +
        df['hotwaterheating'] + df['airconditioning']
    )
    
    df['premium_facilities'] = (
        df['guestroom'] * 2 + df['basement'] * 1.5 + 
        df['airconditioning'] * 1.5 + df['hotwaterheating']
    )
    
    # Location quality (mainroad + prefarea are important)
    df['location_quality'] = df['mainroad'] * 2 + df['prefarea'] * 3
    df['location_x_area'] = df['location_quality'] * df['area']
    df['location_x_facilities'] = df['location_quality'] * df['total_facilities']
    
    # Parking value
    df['has_parking'] = (df['parking'] > 0).astype(int)
    df['parking_squared'] = df['parking'] ** 2
    df['parking_x_area'] = df['parking'] * df['area']
    
    # Property type indicators (derived)
    df['is_large'] = (df['area'] > 6360).astype(int)  # Above 75th percentile
    df['is_luxury'] = ((df['bedrooms'] >= 4) & (df['bathrooms'] >= 2) & 
                       (df['total_facilities'] >= 3)).astype(int)
    df['is_compact'] = ((df['area'] < 3600) & (df['bedrooms'] <= 2)).astype(int)
    
    # Multi-story value
    df['is_multistory'] = (df['stories'] > 1).astype(int)
    df['multistory_x_area'] = df['is_multistory'] * df['area']
    
    # Complex interactions
    df['area_bath_story'] = df['area'] * df['bathrooms'] * df['stories']
    df['quality_index'] = (
        df['area'] * 0.4 + 
        df['bathrooms'] * 500 + 
        df['total_facilities'] * 300 +
        df['parking'] * 200
    )
    
    # Furnishing interactions (furnishing affects value differently by size)
    for furnish_col in [col for col in df.columns if col.startswith('furnish_')]:
        df[f'{furnish_col}_x_area'] = df[furnish_col] * df['area']
        df[f'{furnish_col}_x_quality'] = df[furnish_col] * df['quality_index']
    
    print(f"✓ Created {df.shape[1]} total features")
    return df

def prepare_data_ultimate(df):
    """Prepare with ADVANCED scaling strategy"""
    
    print("\nStep 3: Preparing data with advanced preprocessing...")
    
    # Separate target
    y = df['price']
    X = df.drop('price', axis=1)
    
    # Identify feature types
    binary_features = [col for col in X.columns if X[col].nunique() <= 2]
    
    # For highly skewed features, use QuantileTransformer (better than RobustScaler)
    skewed_features = []
    continuous_features = []
    
    for col in X.columns:
        if col not in binary_features:
            skewness = X[col].skew()
            if abs(skewness) > 1.0:  # Highly skewed
                skewed_features.append(col)
            else:
                continuous_features.append(col)
    
    print(f"✓ Binary features: {len(binary_features)}")
    print(f"✓ Skewed features (use QuantileTransformer): {len(skewed_features)}")
    print(f"✓ Normal features (use RobustScaler): {len(continuous_features)}")
    
    # Apply transformations
    X_transformed = X.copy()
    
    # QuantileTransformer for skewed features (better than log!)
    if skewed_features:
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        X_transformed[skewed_features] = qt.fit_transform(X[skewed_features])
    
    # RobustScaler for normal features
    if continuous_features:
        rs = RobustScaler()
        X_transformed[continuous_features] = rs.fit_transform(X[continuous_features])
    
    # Split data (85-15 split to have more training data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.15, random_state=42
    )
    
    # Log transform target
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    print(f"✓ Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"✓ Applied QuantileTransformer + RobustScaler + Log target")
    
    return X_train, X_test, y_train, y_test, y_train_log, y_test_log, X.columns.tolist()

def train_ultimate_models(X_train, y_train_log, X_test, y_test_log, y_test):
    """Train with ULTIMATE hyperparameters"""
    
    print_section("TRAINING ULTIMATE MODELS")
    
    models = {}
    results = {}
    
    # 1. Ridge (light regularization)
    print("Training Ridge...")
    ridge = Ridge(alpha=0.5, max_iter=10000)
    ridge.fit(X_train, y_train_log)
    models['Ridge'] = ridge
    results['Ridge'] = evaluate(ridge, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Ridge']['r2_score']:.4f}")
    
    # 2. Lasso (very light regularization)
    print("Training Lasso...")
    lasso = Lasso(alpha=0.0001, max_iter=10000)
    lasso.fit(X_train, y_train_log)
    models['Lasso'] = lasso
    results['Lasso'] = evaluate(lasso, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Lasso']['r2_score']:.4f}")
    
    # 3. ElasticNet
    print("Training ElasticNet...")
    elastic = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=10000)
    elastic.fit(X_train, y_train_log)
    models['ElasticNet'] = elastic
    results['ElasticNet'] = evaluate(elastic, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['ElasticNet']['r2_score']:.4f}")
    
    # 4. Random Forest (ULTIMATE config)
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=800,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.5,
        max_samples=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_log)
    models['Random Forest'] = rf
    results['Random Forest'] = evaluate(rf, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Random Forest']['r2_score']:.4f}")
    
    # 5. Gradient Boosting (ULTIMATE config)
    print("Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.9,
        max_features=0.7,
        random_state=42
    )
    gb.fit(X_train, y_train_log)
    models['Gradient Boosting'] = gb
    results['Gradient Boosting'] = evaluate(gb, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Gradient Boosting']['r2_score']:.4f}")
    
    # 6. XGBoost (ULTIMATE config - BEST PERFORMER)
    print("Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        colsample_bylevel=0.9,
        gamma=0.01,
        reg_alpha=0.001,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train_log)
    models['XGBoost'] = xgb
    results['XGBoost'] = evaluate(xgb, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['XGBoost']['r2_score']:.4f}")
    
    # 7. LightGBM (ULTIMATE config - BEST PERFORMER)
    print("Training LightGBM...")
    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=64,
        max_depth=8,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.001,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train, y_train_log)
    models['LightGBM'] = lgbm
    results['LightGBM'] = evaluate(lgbm, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['LightGBM']['r2_score']:.4f}")
    
    # 8. Neural Network
    print("Training Neural Network...")
    y_scaler = StandardScaler()
    y_train_nn = y_scaler.fit_transform(y_train_log.values.reshape(-1, 1)).ravel()
    
    nn = MLPRegressor(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.00001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=150,
        random_state=42
    )
    nn.fit(X_train, y_train_nn)
    models['Neural Network'] = (nn, y_scaler)
    
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
    
    # 9. ENSEMBLE (Voting of top 3 models)
    print("Training Ensemble (Voting)...")
    ensemble = VotingRegressor([
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('gb', gb)
    ], weights=[0.4, 0.4, 0.2])
    ensemble.fit(X_train, y_train_log)
    models['Ensemble'] = ensemble
    results['Ensemble'] = evaluate(ensemble, X_test, y_test_log, y_test)
    print(f"  ✓ R² = {results['Ensemble']['r2_score']:.4f}")
    
    return models, results

def evaluate(model, X_test, y_test_log, y_test_original):
    """Evaluate on original scale"""
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
    return {
        'r2_score': r2_score(y_test_original, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_original, y_pred)),
        'mae': mean_absolute_error(y_test_original, y_pred),
        'mape': np.mean(np.abs((y_test_original - y_pred) / (y_test_original + 1e-10))) * 100
    }

def main():
    print_section("ULTIMATE HOUSING PRICE PREDICTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    df = load_and_preprocess()
    df = engineer_features_ultimate(df)
    X_train, X_test, y_train, y_test, y_train_log, y_test_log, features = prepare_data_ultimate(df)
    
    models, results = train_ultimate_models(X_train, y_train_log, X_test, y_test_log, y_test)
    
    # Display results
    print_section("RESULTS")
    
    results_df = pd.DataFrame(results).T.sort_values('r2_score', ascending=False)
    
    print("┌" + "─" * 78 + "┐")
    print(f"│ {'Model':<20} │ {'R² Score':<10} │ {'RMSE':<15} │ {'MAE':<15} │")
    print("├" + "─" * 78 + "┤")
    
    for name, metrics in results_df.iterrows():
        print(f"│ {name:<20} │ {metrics['r2_score']:<10.4f} │ ₹{metrics['rmse']:<14,.0f} │ ₹{metrics['mae']:<14,.0f} │")
    
    print("└" + "─" * 78 + "┘")
    
    best = results_df.iloc[0]
    best_name = results_df.index[0]
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST MODEL: {best_name.upper()}")
    print(f"   R² = {best['r2_score']:.4f} | RMSE = ₹{best['rmse']:,.0f} | MAE = ₹{best['mae']:,.0f}")
    print(f"{'='*80}")
    
    # Save
    os.makedirs('models/saved_models', exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f'models/saved_models/{name.lower().replace(" ", "_")}.pkl')
    
    with open('models/training_results.json', 'w') as f:
        json.dump({
            'training_date': datetime.now().isoformat(),
            'best_model': best_name,
            'models': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
        }, f, indent=4)
    
    if best['r2_score'] >= 0.85:
        print("\n🎉 EXCELLENT! R² ≥ 0.85!")
    elif best['r2_score'] >= 0.80:
        print("\n✅ GREAT! R² ≥ 0.80!")
    elif best['r2_score'] >= 0.75:
        print("\n👍 GOOD! R² ≥ 0.75!")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()