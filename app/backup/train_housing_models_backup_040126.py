#!/usr/bin/env python3
# train_housing_models.py 

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
import joblib

warnings.filterwarnings('ignore')

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.model_trainer import ModelTrainer

def main():
    print("=" * 80)
    print("HOUSING PRICE PREDICTION - ENHANCED MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Load data
    print("Step 1: Loading Housing.csv...")
    csv_path = 'Housing.csv'
    if not os.path.exists(csv_path):
        print("✗ Housing.csv not found!")
        return

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Price range: ₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}")

    # Step 2: Preprocessing (same as original)
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

    # Step 3: Feature engineering (same as original)
    print("\nStep 3: Feature engineering...")
    #df['price_per_sqft'] = df['price'] / df['area']
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
    df['facilities_score'] = df[['mainroad', 'guestroom', 'basement',
                                 'hotwaterheating', 'airconditioning']].sum(axis=1)
    df['area_category'] = pd.cut(df['area'], bins=[0, 3000, 6000, 10000, 20000],
                                 labels=[0, 1, 2, 3]).astype(int)
    #df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)
    #df['bathroom_density'] = df['bathrooms'] / (df['bedrooms'] + 1)
    #df['rooms_per_story'] = df['bedrooms'] / (df['stories'] + 1)
    #df['amenity_ratio'] = df['facilities_score'] / 5
    print("✓ Created 4 engineered features")

    # Step 4: Clean
    print("\nStep 4: Cleaning data...")
    original_len = len(df)
    df = df.dropna().drop_duplicates()
    print(f"✓ Removed {original_len - len(df)} records")
    print(f"✓ Final dataset: {len(df)} rows")

    # Step 5: Prepare data using ModelTrainer
    print("\nStep 5: Preparing data with ModelTrainer...")
    trainer = ModelTrainer()
    trainer.prepare_data(df, target_col='price', test_size=0.2)

    print(f"✓ Training set: {trainer.X_train.shape}")
    print(f"✓ Test set:     {trainer.X_test.shape}")
    print(f"✓ Features ({len(trainer.feature_names)}): {trainer.feature_names}")

    # Step 6: Train all models (full power - includes Neural Network)
    print("\n" + "=" * 80)
    print("Step 6: Training enhanced models (RandomForest, XGBoost, LightGBM, NN, etc.)")
    print("=" * 80 + "\n")

    metrics_dict = trainer.train_all_models(quick_mode=False)  # Full training

    # Step 7: Display results
    metrics_df = trainer.get_metrics_dataframe()
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(metrics_df[['Model', 'R² Score (Test)', 'RMSE (Test)', 'MAE (Test)', 'Training Time (s)']]\
          .round(4).to_string(index=False))

    best_model_name = trainer.best_model_name
    best_model_name = best_model_name.title()
    best_metrics = metrics_dict[best_model_name]
    print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"   R² (Test): {best_metrics['r2_test']:.4f}")
    print(f"   RMSE:      ₹{best_metrics['rmse_test']:,.0f}")
    print(f"   MAE:       ₹{best_metrics['mae_test']:,.0f}")

    # Step 8: Save models and preprocessing objects (same paths)
    save_dir = 'models/saved_models'
    os.makedirs(save_dir, exist_ok=True)

    print("\nStep 8: Saving models and artifacts...")
    trainer.save_models(directory=save_dir)

    # Save scaler (fit only on training data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(trainer.X_train)
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))

    # Save feature names and train/test splits
    joblib.dump(trainer.feature_names, os.path.join(save_dir, 'feature_names.pkl'))
    joblib.dump(trainer.X_train, os.path.join(save_dir, 'X_train.pkl'))
    joblib.dump(trainer.X_test, os.path.join(save_dir, 'X_test.pkl'))
    joblib.dump(trainer.y_train, os.path.join(save_dir, 'y_train.pkl'))
    joblib.dump(trainer.y_test, os.path.join(save_dir, 'y_test.pkl'))

    print("✓ All models, scaler, and data saved")

    # Step 9: Save training_results.json in ORIGINAL format (with "models" key)
    print("\nStep 9: Saving training_results.json (compatible format)...")
    results_dict = {
        'training_date': datetime.now().isoformat(),
        'dataset': 'Housing.csv',
        'total_records': len(df),
        'training_records': len(trainer.X_train),
        'test_records': len(trainer.X_test),
        'features': trainer.feature_names,
        'best_model': best_model_name,
        'models': {}
    }

    for model_name, m in metrics_dict.items():
        # Predict on test set
        if model_name == 'Neural Network':
            y_pred_test = m['model'].predict(trainer.X_test, verbose=0).flatten()
        else:
            y_pred_test = m['model'].predict(trainer.X_test)

        # Compute MAPE safely
        y_true = trainer.y_test.values
        y_pred = y_pred_test
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        results_dict['models'][model_name] = {
            'rmse': float(m['rmse_test']),
            'mae': float(m['mae_test']),
            'r2_score': float(m['r2_test']),
            'mape': float(mape)
        }

    with open('models/training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    # Step 10: Save model_state.json (unchanged)
    model_state = {
        'trained': True,
        'feature_names': trainer.feature_names,
        'best_model': best_model_name,
        'model_count': len(trainer.models),
        'training_date': datetime.now().isoformat()
    }
    with open('models/model_state.json', 'w') as f:
        json.dump(model_state, f, indent=4)

    print("✓ training_results.json and model_state.json saved (compatible)")

    # Step 11: Sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (Best Model)")
    print("=" * 80 + "\n")

    indices = np.random.choice(len(trainer.X_test), size=min(5, len(trainer.X_test)), replace=False)
    test_df = trainer.X_test.reset_index(drop=True)
    y_test_reset = trainer.y_test.reset_index(drop=True)

    for i, idx in enumerate(indices, 1):
        sample = test_df.iloc[[idx]]
        actual = y_test_reset.iloc[idx]
        predicted = trainer.predict(sample)[0]

        orig_idx = trainer.y_test.index[idx]
        area = df.loc[orig_idx, 'area']
        beds = df.loc[orig_idx, 'bedrooms']
        baths = df.loc[orig_idx, 'bathrooms']

        error = predicted - actual
        error_pct = (error / actual) * 100 if actual != 0 else 0

        print(f"Property {i}:")
        print(f"  Area: {int(area)} sq ft | Beds: {int(beds)} | Baths: {int(baths)}")
        print(f"  Actual:    ₹{actual:,.0f}")
        print(f"  Predicted: ₹{predicted:,.0f}")
        print(f"  Error:     ₹{error:,.0f} ({error_pct:+.1f}%)\n")

    print("=" * 80)
    print("✅ TRAINING COMPLETE - ENHANCED MODELS READY!")
    print("=" * 80)
    print("\n📦 All files saved in original format → Streamlit app will load perfectly")
    print(f"🏆 Best model: {best_model_name}")
    print("\n🚀 Next: streamlit run app/streamlit_app.py")
    print("=" * 80)

if __name__ == "__main__":
    main()