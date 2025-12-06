#!/usr/bin/env python3
# train_housing_models.py

"""
Training script specifically for Housing.csv dataset
Trains all models and saves them for the Streamlit app
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
        print(f"✓ Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
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
    y = df['price']
    
    # Store feature names
    feature_names = X.columns.tolist()
    print(f"✓ Features: {feature_names}")
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # Save preprocessor objects
    print("\nStep 6: Saving preprocessing objects...")
    os.makedirs('models/saved_models', exist_ok=True)
    
    import joblib
    joblib.dump(scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(feature_names, 'models/saved_models/feature_names.pkl')
    print("✓ Saved scaler and feature names")
    
    # Step 7: Train all models
    print("\n" + "=" * 80)
    print("Step 7: Training models...")
    print("=" * 80 + "\n")
    
    results = models.train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 8: Display results
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80 + "\n")
    
    # Create results table
    results_data = []
    for model_name, metrics in results.items():
        results_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'RMSE': f"${metrics['rmse']:,.0f}",
            'MAE': f"${metrics['mae']:,.0f}",
            'R² Score': f"{metrics['r2_score']:.4f}",
            'MAPE': f"{metrics['mape']:.2f}%"
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('R² Score', ascending=False)
    print(results_df.to_string(index=False))
    
    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MODEL: {models.best_model_name.upper()}")
    print(f"   R² Score: {results[models.best_model_name]['r2_score']:.4f}")
    print(f"   RMSE: ${results[models.best_model_name]['rmse']:,.0f}")
    print(f"   MAE: ${results[models.best_model_name]['mae']:,.0f}")
    print(f"{'=' * 80}")
    
    # Step 9: Save models
    print("\nStep 8: Saving trained models...")
    models.save_models('models/saved_models/')
    print("✓ All models saved")
    
    # Step 10: Save results
    print("\nStep 9: Saving training results...")
    
    results_dict = {
        'training_date': datetime.now().isoformat(),
        'dataset': 'Housing.csv',
        'total_records': len(df),
        'training_records': len(X_train),
        'test_records': len(X_test),
        'features': feature_names,
        'best_model': models.best_model_name,
        'models': {}
    }
    
    for model_name, metrics in results.items():
        results_dict['models'][model_name] = {
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'r2_score': float(metrics['r2_score']),
            'mape': float(metrics['mape'])
        }
    
    with open('models/training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print("✓ Results saved to models/training_results.json")
    
    # Step 11: Generate sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80 + "\n")
    
    # Get 5 random test samples
    indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(indices, 1):
        actual = y_test.iloc[idx]
        predicted = models.predict(X_test.iloc[[idx]])[0]
        error = predicted - actual
        error_pct = (error / actual) * 100
        
        print(f"Property {i}:")
        print(f"  Area: {int(df.iloc[y_test.index[idx]]['area'])} sq ft")
        print(f"  Bedrooms: {int(df.iloc[y_test.index[idx]]['bedrooms'])}")
        print(f"  Bathrooms: {int(df.iloc[y_test.index[idx]]['bathrooms'])}")
        print(f"  Actual Price:    ${actual:,.0f}")
        print(f"  Predicted Price: ${predicted:,.0f}")
        print(f"  Error:           ${error:,.0f} ({error_pct:+.1f}%)")
        print()
    
    # Final summary
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Models saved in: models/saved_models/")
    print(f"✓ Best model: {models.best_model_name}")
    print(f"✓ Ready for deployment!")
    print(f"\nNext step: Run the Streamlit app with:")
    print(f"  streamlit run app/streamlit_app.py")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()