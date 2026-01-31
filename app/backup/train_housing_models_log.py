#!/usr/bin/env python3
# train_housing_models.py - CORRECTED VERSION

"""
Training script for Housing.csv dataset
Trains all models and saves them for the Streamlit app
Compatible with dynamic Model Insights page
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import RealEstateDataPreprocessor
from src.predictive_models import RealEstatePredictiveModels

def main():
    print("=" * 80)
    print("HOUSING PRICE PREDICTION - MODEL TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize components
    preprocessor = RealEstateDataPreprocessor()
    models = RealEstatePredictiveModels()
    
    # Step 1: Load Housing.csv
    print("Step 1: Loading Housing.csv...")
    try:
        df = pd.read_csv('Housing.csv')
        print(f"✓ Loaded {len(df)} records")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ Price range: ₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}")
    except FileNotFoundError:
        print("✗ Housing.csv not found in current directory!")
        print("Please ensure Housing.csv is in the same directory as this script.")
        return
    
    # Step 2: Process housing-specific data
    print("\nStep 2: Processing housing data...")
    
    # Convert yes/no to binary
    binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                     'airconditioning', 'prefarea']
    
    for col in binary_columns:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    
    # Map furnishing status
    df['furnishingstatus'] = df['furnishingstatus'].map({
        'furnished': 2,
        'semi-furnished': 1,
        'unfurnished': 0
    })
    
    print("✓ Converted binary columns")
    print("✓ Encoded furnishing status")
    
    # Step 3: Feature Engineering
    print("\nStep 3: Feature engineering...")
    
    # Price per square foot
    df['price_per_sqft'] = df['price'] / df['area']
    
    # Bed-bath ratio
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
    
    # Total facilities score
    df['facilities_score'] = (df['mainroad'] + df['guestroom'] + df['basement'] + 
                              df['hotwaterheating'] + df['airconditioning'])
    
    # Area category
    df['area_category'] = pd.cut(df['area'], bins=[0, 3000, 6000, 10000, 20000], 
                                 labels=[0, 1, 2, 3])
    df['area_category'] = df['area_category'].astype(int)
    
    print("✓ Created 4 new features")
    print(f"✓ Total features: {df.shape[1]}")
    
    # Step 4: Clean data
    print("\nStep 4: Cleaning data...")
    original_count = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"✓ Removed {original_count - len(df)} invalid records")
    print(f"✓ Final dataset: {len(df)} records")
    
    # Step 5: Prepare features
    print("\nStep 5: Preparing features for modeling...")
    
    # Separate features and target
    X = df.drop(columns=['price', 'price_per_sqft'])
    y = np.log1p(df['price'])
    
    # Store feature names
    feature_names = X.columns.tolist()
    print(f"✓ Features: {feature_names}")
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Train-test split - CORRECTED IMPORT
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # Step 6: Save preprocessing objects
    print("\nStep 6: Saving preprocessing objects...")
    os.makedirs('models/saved_models', exist_ok=True)
    
    import joblib
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(feature_names, 'models/saved_models/feature_names.pkl')
    
    # Save train/test data for Model Insights explainability
    joblib.dump(X_train, 'models/saved_models/X_train.pkl')
    joblib.dump(X_test, 'models/saved_models/X_test.pkl')
    joblib.dump(y_train, 'models/saved_models/y_train.pkl')
    joblib.dump(y_test, 'models/saved_models/y_test.pkl')
    
    print("✓ Saved scaler and feature names")
    print("✓ Saved train/test data for explainability")
    
    # Step 7: Train all models
    print("\n" + "=" * 80)
    print("Step 7: Training all 7 models...")
    print("=" * 80 + "\n")
    
    results = models.train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 8: Display results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80 + "\n")
    
    # Create results table using model_metrics (display names)
    results_data = []
    
    if hasattr(models, 'model_metrics') and models.model_metrics:
        for model_name, metrics in models.model_metrics.items():
            results_data.append({
                'Model': model_name,
                'RMSE': f"₹{metrics.get('rmse', 0):,.0f}",
                'MAE': f"₹{metrics.get('mae', 0):,.0f}",
                'R² Score': f"{metrics.get('r2', metrics.get('r2_score', 0)):.4f}",
                'MAPE': f"{metrics.get('mape', 0):.2f}%"
            })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('R² Score', ascending=False)
    
    # Pretty print table
    print("┌" + "─" * 78 + "┐")
    print(f"│ {'Model':<25} │ {'RMSE':<15} │ {'MAE':<15} │ {'R² Score':<10} │")
    print("├" + "─" * 78 + "┤")
    
    for _, row in results_df.iterrows():
        print(f"│ {row['Model']:<25} │ {row['RMSE']:<15} │ {row['MAE']:<15} │ {row['R² Score']:<10} │")
    
    print("└" + "─" * 78 + "┘")
    
    # Display best model
    best_model_display = models.get_best_model()
    best_metrics = models.model_metrics[best_model_display]
    
    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MODEL: {best_model_display.upper()}")
    print(f"   R² Score: {best_metrics.get('r2', best_metrics.get('r2_score', 0)):.4f}")
    print(f"   RMSE: ₹{best_metrics.get('rmse', 0):,.0f}")
    print(f"   MAE: ₹{best_metrics.get('mae', 0):,.0f}")
    print(f"{'=' * 80}")
    
    # Step 9: Save models
    print("\nStep 8: Saving trained models...")
    models.save_models('models/saved_models/')
    print("✓ All 7 models saved successfully")
    
    # Step 10: Save comprehensive results
    print("\nStep 9: Saving training results...")
    
    results_dict = {
        'training_date': datetime.now().isoformat(),
        'dataset': 'Housing.csv',
        'total_records': len(df),
        'training_records': len(X_train),
        'test_records': len(X_test),
        'features': feature_names,
        'best_model': best_model_display,
        'models': {}
    }
    
    # Save results with display names
    for model_name, metrics in models.model_metrics.items():
        results_dict['models'][model_name] = {
            'rmse': float(np.expm1(metrics.get('rmse', 0))),
            'mae': float(metrics.get('mae', 0)),
            'r2_score': float(metrics.get('r2', metrics.get('r2_score', 0))),
            'mape': float(metrics.get('mape', 0))
        }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print("✓ Results saved to models/training_results.json")
    
    # Save model state for Streamlit quick-load
    model_state = {
        'trained': True,
        'feature_names': feature_names,
        'best_model': best_model_display,
        'model_count': len(models.models),
        'training_date': datetime.now().isoformat()
    }
    
    with open('models/model_state.json', 'w') as f:
        json.dump(model_state, f, indent=4)
    
    print("✓ Model state saved for Streamlit quick-load")
    
    # Step 11: Generate sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80 + "\n")
    
    # Get 5 random test samples
    indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for i, idx in enumerate(indices, 1):
        actual = np.expm1(y_test.iloc[idx])
        predicted = np.expm1(models.predict(X_test.iloc[[idx]])[0])
        error = predicted - actual
        error_pct = (error / actual) * 100
        
        # Get original features for display
        original_idx = y_test.index[idx]
        
        print(f"Property {i}:")
        print(f"  Area: {int(df.loc[original_idx, 'area'])} sq ft")
        print(f"  Bedrooms: {int(df.loc[original_idx, 'bedrooms'])}")
        print(f"  Bathrooms: {int(df.loc[original_idx, 'bathrooms'])}")
        print(f"  Actual Price:    ₹{actual:,.0f}")
        print(f"  Predicted Price: ₹{predicted:,.0f}")
        print(f"  Error:           ₹{error:,.0f} ({error_pct:+.1f}%)")
        print()
    
    # Final summary
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📦 Files saved:")
    print(f"   ✓ models/saved_models/ - All 7 models + training data")
    print(f"   ✓ models/training_results.json - Performance metrics")
    print(f"   ✓ models/model_state.json - Quick-load state")
    print(f"\n🏆 Best model: {best_model_display}")
    print(f"💾 Training data saved for explainability")
    print(f"⚡ Model state saved for quick loading")
    print(f"✨ Ready for deployment!")
    print(f"\n📊 Model Insights Integration Status:")
    print(f"   ✓ Dynamic feature importance ready")
    print(f"   ✓ Real-time performance metrics ready")
    print(f"   ✓ SHAP explainability data ready")
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run: streamlit run app/streamlit_app.py")
    print(f"   2. Navigate to '📊 Model Insights'")
    print(f"   3. Models should load automatically!")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()