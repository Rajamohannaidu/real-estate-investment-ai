#!/usr/bin/env python3
# train_housing_CORRECTED.py - FIXED VERSION

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings('ignore')

print("=" * 80)
print("HOUSING PRICE PREDICTION - CORRECTED TRAINING")
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

# Step 2: Preprocessing
print("\nStep 2: Preprocessing categorical columns...")
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
               'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df['furnishingstatus'] = df['furnishingstatus'].map({
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
})
print("✓ Encoded binary and furnishing columns")

# Step 3: Feature engineering - REMOVE price_per_sqft (DATA LEAKAGE!)
print("\nStep 3: Feature engineering (CORRECTED - no data leakage)...")

# ✓ SAFE features (don't contain price)
df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
df['facilities_score'] = df[['mainroad', 'guestroom', 'basement',
                             'hotwaterheating', 'airconditioning']].sum(axis=1)
df['area_category'] = pd.cut(df['area'], bins=[0, 3000, 6000, 10000, 20000],
                             labels=[0, 1, 2, 3]).astype(int)

# ❌ REMOVED: price_per_sqft (contains target variable!)
print("✓ Created 3 engineered features (removed price_per_sqft)")

# Step 4: Clean
print("\nStep 4: Cleaning data...")
original_len = len(df)
df = df.dropna().drop_duplicates()
print(f"✓ Removed {original_len - len(df)} records")
print(f"✓ Final dataset: {len(df)} rows")

# Step 5: Prepare features
print("\nStep 5: Preparing features and target...")

# Separate features and target
X = df.drop(columns=['price'])
y = df['price']

# Get feature names
feature_names = X.columns.tolist()
print(f"✓ Features ({len(feature_names)}): {feature_names}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set:     {X_test.shape}")

# Step 6: Scale features
print("\nStep 6: Scaling features...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),  # ← FIT on train only!
    columns=feature_names,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),  # ← TRANSFORM test
    columns=feature_names,
    index=X_test.index
)
print("✓ Features scaled")

# Step 7: Train models
print("\n" + "=" * 80)
print("Step 7: Training Models")
print("=" * 80 + "\n")

models = {}
metrics = {}

# 1. Linear Regression
print("Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
models['Linear Regression'] = lr

y_pred_lr = lr.predict(X_test_scaled)
metrics['Linear Regression'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'mae': mean_absolute_error(y_test, y_pred_lr),
    'r2_score': r2_score(y_test, y_pred_lr),
    'mape': np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['Linear Regression']['r2_score']:.4f}")

# 2. Ridge
print("Training Ridge...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
models['Ridge'] = ridge

y_pred_ridge = ridge.predict(X_test_scaled)
metrics['Ridge'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    'mae': mean_absolute_error(y_test, y_pred_ridge),
    'r2_score': r2_score(y_test, y_pred_ridge),
    'mape': np.mean(np.abs((y_test - y_pred_ridge) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['Ridge']['r2_score']:.4f}")

# 3. Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
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

# 4. XGBoost
print("Training XGBoost...")
xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
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

# 5. LightGBM
print("Training LightGBM...")
lgbm = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, 
                     random_state=42, n_jobs=-1, verbose=-1)
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

# 6. Gradient Boosting
print("Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
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

# 7. Neural Network
print("Training Neural Network...")
nn = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])

nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

nn.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, 
       callbacks=[early_stop], verbose=0)

models['Neural Network'] = nn

y_pred_nn = nn.predict(X_test_scaled, verbose=0).flatten()
metrics['Neural Network'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
    'mae': mean_absolute_error(y_test, y_pred_nn),
    'r2_score': r2_score(y_test, y_pred_nn),
    'mape': np.mean(np.abs((y_test - y_pred_nn) / y_test)) * 100
}
print(f"  ✓ R² = {metrics['Neural Network']['r2_score']:.4f}")

# Find best model
best_model_name = max(metrics.items(), key=lambda x: x[1]['r2_score'])[0]
best_model = models[best_model_name]
print(f"\n🏆 BEST MODEL: {best_model_name} (R² = {metrics[best_model_name]['r2_score']:.4f})")

# Step 8: Display results
print("\n" + "=" * 80)
print("TRAINING RESULTS")
print("=" * 80)

results_df = pd.DataFrame(metrics).T
results_df = results_df.round(4).sort_values('r2_score', ascending=False)
print(results_df)

# Step 9: Save models
save_dir = 'models/saved_models'
os.makedirs(save_dir, exist_ok=True)

print("\nStep 8: Saving models and artifacts...")

# Save each model
for name, model in models.items():
    if name == 'Neural Network':
        model.save(os.path.join(save_dir, 'neural_network.h5'))
    else:
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, os.path.join(save_dir, filename))

# Save scaler
joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

# Save feature names
joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.pkl'))

# Save train/test data
joblib.dump(X_train_scaled, os.path.join(save_dir, 'X_train.pkl'))
joblib.dump(X_test_scaled, os.path.join(save_dir, 'X_test.pkl'))
joblib.dump(y_train, os.path.join(save_dir, 'y_train.pkl'))
joblib.dump(y_test, os.path.join(save_dir, 'y_test.pkl'))

print("✓ Models, scaler, and data saved")

# Step 10: Save training_results.json
results_dict = {
    'training_date': datetime.now().isoformat(),
    'dataset': 'Housing.csv',
    'total_records': len(df),
    'training_records': len(X_train),
    'test_records': len(X_test),
    'features': feature_names,
    'best_model': best_model_name,
    'models': {name: {k: float(v) for k, v in m.items()} for name, m in metrics.items()}
}

with open('models/training_results.json', 'w') as f:
    json.dump(results_dict, f, indent=4)

# Save model_state.json
model_state = {
    'trained': True,
    'feature_names': feature_names,
    'best_model': best_model_name,
    'model_count': len(models),
    'training_date': datetime.now().isoformat()
}

with open('models/model_state.json', 'w') as f:
    json.dump(model_state, f, indent=4)

print("✓ training_results.json and model_state.json saved")

# Step 11: Test prediction
print("\n" + "=" * 80)
print("TEST PREDICTION (5000 sq ft, 3 bed, 2 bath property)")
print("=" * 80)

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
    'area_category': 2
}

test_df = pd.DataFrame([test_property])
test_df = test_df[feature_names]  # Ensure correct order
test_scaled = scaler.transform(test_df)

# Predict with best model
if best_model_name == 'Neural Network':
    prediction = best_model.predict(test_scaled, verbose=0).flatten()[0]
else:
    prediction = best_model.predict(test_scaled)[0]

print(f"\n🏠 Property Details:")
print(f"  • Area: 5000 sq ft")
print(f"  • Bedrooms: 3")
print(f"  • Bathrooms: 2")
print(f"  • All amenities: Yes")

print(f"\n💰 Predicted Price: ₹{prediction:,.0f}")

# Compare with dataset
similar = df[
    (df['area'] >= 4500) & (df['area'] <= 5500) &
    (df['bedrooms'] == 3) &
    (df['bathrooms'] == 2)
]

if len(similar) > 0:
    print(f"\n📊 Similar Properties in Dataset:")
    print(f"  • Average: ₹{similar['price'].mean():,.0f}")
    print(f"  • Range: ₹{similar['price'].min():,.0f} - ₹{similar['price'].max():,.0f}")
    
    error_pct = abs(prediction - similar['price'].mean()) / similar['price'].mean() * 100
    if error_pct < 10:
        print(f"  ✅ Prediction within 10% of average ({error_pct:.1f}%)")
    elif error_pct < 20:
        print(f"  ⚠️ Prediction within 20% of average ({error_pct:.1f}%)")
    else:
        print(f"  ❌ Prediction differs by {error_pct:.1f}% from average")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE - CORRECTED MODELS READY!")
print("=" * 80)
print(f"\n🏆 Best model: {best_model_name}")
print(f"📦 All files saved in models/saved_models/")
print(f"\n🚀 Next: streamlit run streamlit_app.py")
print("=" * 80)