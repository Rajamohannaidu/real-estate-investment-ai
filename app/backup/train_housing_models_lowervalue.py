#!/usr/bin/env python3
# train_housing_ADVANCED.py - Improved Performance with Advanced Techniques

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings('ignore')

print("=" * 80)
print("HOUSING PRICE PREDICTION - ADVANCED MODEL TRAINING")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Step 1: Load data
print("Step 1: Loading Housing.csv...")
csv_path = 'Housing.csv'
if not os.path.exists(csv_path):
    print("❌ Housing.csv not found!")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"✓ Loaded {len(df)} records")
print(f"✓ Price range: ₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}")

# Step 2: Enhanced Preprocessing
print("\nStep 2: Enhanced preprocessing...")
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
               'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df['furnishingstatus'] = df['furnishingstatus'].map({
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
})
print("✓ Encoded categorical features")

# Step 3: ADVANCED Feature Engineering
print("\nStep 3: Advanced feature engineering...")

# Basic ratios
df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
df['facilities_score'] = df[['mainroad', 'guestroom', 'basement',
                             'hotwaterheating', 'airconditioning']].sum(axis=1)

# NEW: More sophisticated features
df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)
df['area_per_bathroom'] = df['area'] / (df['bathrooms'] + 1)
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['room_density'] = df['total_rooms'] / df['stories']
df['luxury_score'] = (df['furnishingstatus'] * 2 + 
                      df['airconditioning'] * 3 + 
                      df['parking'] * 1.5 + 
                      df['basement'] * 2)

# Area categories with more granularity
df['area_small'] = (df['area'] < 3000).astype(int)
df['area_medium'] = ((df['area'] >= 3000) & (df['area'] < 6000)).astype(int)
df['area_large'] = ((df['area'] >= 6000) & (df['area'] < 10000)).astype(int)
df['area_very_large'] = (df['area'] >= 10000).astype(int)

# Polynomial features (area squared for non-linear relationships)
df['area_squared'] = df['area'] ** 2
df['area_log'] = np.log1p(df['area'])

# Interaction features
df['bed_area_interaction'] = df['bedrooms'] * df['area']
df['bath_area_interaction'] = df['bathrooms'] * df['area']
df['parking_area'] = df['parking'] * df['area']

print(f"✓ Created {df.shape[1] - 13} new features")  # Original had 13 columns

# Step 4: Remove outliers (IQR method)
print("\nStep 4: Removing outliers...")
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

original_len = len(df)
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
print(f"✓ Removed {original_len - len(df)} outliers")
print(f"✓ Final dataset: {len(df)} rows")

# Step 5: Prepare features
print("\nStep 5: Preparing features and target...")
X = df.drop(columns=['price'])
y = df['price']

feature_names = X.columns.tolist()
print(f"✓ Total features: {len(feature_names)}")

# Step 6: Train-test split with stratification
print("\nStep 6: Splitting data (stratified)...")

# Create bins for stratified split
price_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=price_bins
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set:     {X_test.shape}")

# Step 7: Advanced scaling (RobustScaler - better for outliers)
print("\nStep 7: Scaling with RobustScaler...")
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names,
    index=X_test.index
)
print("✓ Features scaled with RobustScaler")

# Step 8: Train OPTIMIZED models
print("\n" + "=" * 80)
print("Step 8: Training Optimized Models")
print("=" * 80 + "\n")

models = {}
metrics = {}

# 1. Ridge with GridSearch
print("Training Ridge (with hyperparameter tuning)...")
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2', n_jobs=-1)
ridge_grid.fit(X_train_scaled, y_train)
models['Ridge'] = ridge_grid.best_estimator_

y_pred_ridge = ridge_grid.predict(X_test_scaled)
metrics['Ridge'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'mae': mean_absolute_error(y_test, y_pred_ridge),
    'r2_score': r2_score(y_test, y_pred_ridge),
    'mape': np.mean(np.abs((y_test - y_pred_ridge) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['Ridge']['r2_score']:.4f} (alpha={ridge_grid.best_params_['alpha']})")

# 2. Random Forest (optimized hyperparameters)
print("Training Random Forest (optimized)...")
rf = RandomForestRegressor(
    n_estimators=300,          # More trees
    max_depth=20,              # Deeper trees
    min_samples_split=5,       # Prevent overfitting
    min_samples_leaf=2,
    max_features='sqrt',       # Feature subsampling
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf

y_pred_rf = rf.predict(X_test_scaled)
metrics['Random Forest'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'r2_score': r2_score(y_test, y_pred_rf),
    'mape': np.mean(np.abs((y_test - y_pred_rf) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['Random Forest']['r2_score']:.4f}")

# 3. XGBoost (optimized hyperparameters)
print("Training XGBoost (optimized)...")
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=8,               # Deeper for complex patterns
    learning_rate=0.05,        # Lower learning rate
    subsample=0.8,             # Row sampling
    colsample_bytree=0.8,      # Column sampling
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,             # L1 regularization
    reg_lambda=1.0,            # L2 regularization
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_scaled, y_train)
models['XGBoost'] = xgb

y_pred_xgb = xgb.predict(X_test_scaled)
metrics['XGBoost'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'mae': mean_absolute_error(y_test, y_pred_xgb),
    'r2_score': r2_score(y_test, y_pred_xgb),
    'mape': np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['XGBoost']['r2_score']:.4f}")

# 4. LightGBM (optimized hyperparameters)
print("Training LightGBM (optimized)...")
lgbm = LGBMRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgbm.fit(X_train_scaled, y_train)
models['LightGBM'] = lgbm

y_pred_lgbm = lgbm.predict(X_test_scaled)
metrics['LightGBM'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lgbm)),
    'mae': mean_absolute_error(y_test, y_pred_lgbm),
    'r2_score': r2_score(y_test, y_pred_lgbm),
    'mape': np.mean(np.abs((y_test - y_pred_lgbm) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['LightGBM']['r2_score']:.4f}")

# 5. Gradient Boosting (optimized)
print("Training Gradient Boosting (optimized)...")
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb

y_pred_gb = gb.predict(X_test_scaled)
metrics['Gradient Boosting'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
    'mae': mean_absolute_error(y_test, y_pred_gb),
    'r2_score': r2_score(y_test, y_pred_gb),
    'mape': np.mean(np.abs((y_test - y_pred_gb) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['Gradient Boosting']['r2_score']:.4f}")

# 6. Improved Neural Network
print("Training Improved Neural Network...")

# Add batch normalization and better architecture
nn = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

nn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001
)

nn.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

models['Neural Network'] = nn

y_pred_nn = nn.predict(X_test_scaled, verbose=0).flatten()
metrics['Neural Network'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
    'mae': mean_absolute_error(y_test, y_pred_nn),
    'r2_score': r2_score(y_test, y_pred_nn),
    'mape': np.mean(np.abs((y_test - y_pred_nn) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['Neural Network']['r2_score']:.4f}")

# 7. NEW: Stacking Ensemble (combines best models)
print("Training Stacking Ensemble (meta-model)...")

base_models = [
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ('xgb', XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)),
    ('lgbm', LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1))
]

stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

stacking.fit(X_train_scaled, y_train)
models['Stacking Ensemble'] = stacking

y_pred_stack = stacking.predict(X_test_scaled)
metrics['Stacking Ensemble'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_stack)),
    'mae': mean_absolute_error(y_test, y_pred_stack),
    'r2_score': r2_score(y_test, y_pred_stack),
    'mape': np.mean(np.abs((y_test - y_pred_stack) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['Stacking Ensemble']['r2_score']:.4f}")

# Find best model
best_model_name = max(metrics.items(), key=lambda x: x[1]['r2_score'])[0]
best_model = models[best_model_name]
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   R² = {metrics[best_model_name]['r2_score']:.4f}")
print(f"   RMSE = ₹{metrics[best_model_name]['rmse']:,.0f}")
print(f"   MAE = ₹{metrics[best_model_name]['mae']:,.0f}")

# Display results
print("\n" + "=" * 80)
print("TRAINING RESULTS (Sorted by R²)")
print("=" * 80)

results_df = pd.DataFrame(metrics).T
results_df = results_df.round(4).sort_values('r2_score', ascending=False)
print(results_df.to_string())

# Compare with previous results
print("\n" + "=" * 80)
print("IMPROVEMENT vs PREVIOUS")
print("=" * 80)
previous_best_r2 = 0.6448  # Previous Random Forest
current_best_r2 = metrics[best_model_name]['r2_score']
improvement = ((current_best_r2 - previous_best_r2) / previous_best_r2) * 100

print(f"\nPrevious best R²: {previous_best_r2:.4f}")
print(f"Current best R²:  {current_best_r2:.4f}")
print(f"Improvement:      {improvement:+.2f}%")

# Save models
save_dir = 'models/saved_models'
os.makedirs(save_dir, exist_ok=True)

print("\n" + "=" * 80)
print("Saving Models and Artifacts")
print("=" * 80)

for name, model in models.items():
    if name == 'Neural Network':
        model.save(os.path.join(save_dir, 'neural_network.h5'))
    else:
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, os.path.join(save_dir, filename))

joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.pkl'))
joblib.dump(X_train_scaled, os.path.join(save_dir, 'X_train.pkl'))
joblib.dump(X_test_scaled, os.path.join(save_dir, 'X_test.pkl'))
joblib.dump(y_train, os.path.join(save_dir, 'y_train.pkl'))
joblib.dump(y_test, os.path.join(save_dir, 'y_test.pkl'))

print("✓ All models and artifacts saved")

# Save results
results_dict = {
    'training_date': datetime.now().isoformat(),
    'dataset': 'Housing.csv',
    'total_records': len(df),
    'training_records': len(X_train),
    'test_records': len(X_test),
    'features': feature_names,
    'num_features': len(feature_names),
    'best_model': best_model_name,
    'improvement_over_previous': f"{improvement:+.2f}%",
    'models': {name: {k: float(v) for k, v in m.items()} for name, m in metrics.items()}
}

with open('models/training_results.json', 'w') as f:
    json.dump(results_dict, f, indent=4)

model_state = {
    'trained': True,
    'feature_names': feature_names,
    'best_model': best_model_name,
    'model_count': len(models),
    'training_date': datetime.now().isoformat()
}

with open('models/model_state.json', 'w') as f:
    json.dump(model_state, f, indent=4)

print("✓ Results saved to training_results.json")

# Test prediction
print("\n" + "=" * 80)
print("TEST PREDICTION (5000 sq ft, 3 bed, 2 bath)")
print("=" * 80)

# Recreate all features for test property
test_property = {
    'area': 5000,
    'bedrooms': 3,
    'bathrooms': 2,
    'stories': 2,
    'mainroad': 1,
    'guestroom': 1,
    'basement': 1,
    'hotwaterheating': 1,
    'airconditioning': 1,
    'parking': 2,
    'prefarea': 1,
    'furnishingstatus': 2,
    'bed_bath_ratio': 3 / (2 + 1),
    'facilities_score': 5,
    'area_per_bedroom': 5000 / (3 + 1),
    'area_per_bathroom': 5000 / (2 + 1),
    'total_rooms': 3 + 2,
    'room_density': 5 / 2,
    'luxury_score': (2 * 2 + 1 * 3 + 2 * 1.5 + 1 * 2),
    'area_small': 0,
    'area_medium': 0,
    'area_large': 1,
    'area_very_large': 0,
    'area_squared': 5000 ** 2,
    'area_log': np.log1p(5000),
    'bed_area_interaction': 3 * 5000,
    'bath_area_interaction': 2 * 5000,
    'parking_area': 2 * 5000
}

test_df = pd.DataFrame([test_property])
test_df = test_df[feature_names]
test_scaled = scaler.transform(test_df)

if best_model_name == 'Neural Network':
    prediction = best_model.predict(test_scaled, verbose=0).flatten()[0]
else:
    prediction = best_model.predict(test_scaled)[0]

print(f"\n💰 Predicted Price: ₹{prediction:,.0f}")
print(f"   ({prediction/100000:.2f} lakhs)")

similar = df[(df['area'] >= 4500) & (df['area'] <= 5500) & 
             (df['bedrooms'] == 3) & (df['bathrooms'] == 2)]

if len(similar) > 0:
    print(f"\n📊 Similar Properties:")
    print(f"   Average: ₹{similar['price'].mean():,.0f}")
    error_pct = abs(prediction - similar['price'].mean()) / similar['price'].mean() * 100
    print(f"   Error: {error_pct:.1f}%")
    
    if error_pct < 5:
        print(f"   ✅ Excellent prediction!")
    elif error_pct < 10:
        print(f"   ✅ Good prediction!")
    elif error_pct < 15:
        print(f"   ⚠️ Acceptable prediction")
    else:
        print(f"   ⚠️ Review needed")

print("\n" + "=" * 80)
print("✅ ADVANCED TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🎯 Key Improvements:")
print(f"   • {len(feature_names)} features (was 15)")
print(f"   • RobustScaler (better for outliers)")
print(f"   • Outlier removal")
print(f"   • Optimized hyperparameters")
print(f"   • Stacking ensemble")
print(f"   • Improved neural network")
print(f"\n🏆 Best: {best_model_name} - R² {current_best_r2:.4f}")
print(f"📈 Improvement: {improvement:+.2f}%")
print(f"\n🚀 Next: streamlit run streamlit_app.py")
print("=" * 80)