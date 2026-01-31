#!/usr/bin/env python3
# train_housing_models_improved.py - COMPLETE IMPROVED VERSION

"""
Improved training script for Housing.csv dataset
Expected performance: R² > 0.80 for best models
Includes all optimizations and fixes
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

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

def print_subsection(title):
    """Print formatted subsection"""
    print(f"\n{title}")
    print("-" * len(title))

def main():
    print_section("IMPROVED HOUSING PRICE PREDICTION - MODEL TRAINING")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize components
    preprocessor = RealEstateDataPreprocessor()
    models = RealEstatePredictiveModels()
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print_subsection("Step 1: Loading Housing.csv")
    
    try:
        df = preprocessor.load_data('Housing.csv')
        print(f"Columns: {list(df.columns)}")
        print(f"Price range: ₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}")
        print(f"Mean price: ₹{df['price'].mean():,.0f}")
    except FileNotFoundError:
        print("❌ Housing.csv not found in current directory!")
        print("Please ensure Housing.csv is in the same directory as this script.")
        return
    
    # =========================================================================
    # STEP 2: Process Housing-Specific Features
    # =========================================================================
    print_subsection("Step 2: Processing housing-specific features")
    
    df = preprocessor.process_housing_data(df)
    print("✓ Converted binary columns (yes/no → 1/0)")
    print("✓ One-hot encoded furnishing status")
    
    # =========================================================================
    # STEP 3: Remove Outliers
    # =========================================================================
    print_subsection("Step 3: Removing outliers")
    
    original_count = len(df)
    df = preprocessor.remove_outliers(df, columns=['price', 'area'])
    
    print(f"Records before: {original_count}")
    print(f"Records after: {len(df)}")
    print(f"Records removed: {original_count - len(df)}")
    
    # =========================================================================
    # STEP 4: Clean Data
    # =========================================================================
    print_subsection("Step 4: Cleaning data")
    
    df = preprocessor.clean_data(df)
    print("✓ Handled missing values")
    print("✓ Removed duplicates")
    
    # =========================================================================
    # STEP 5: Advanced Feature Engineering
    # =========================================================================
    print_subsection("Step 5: Advanced feature engineering")
    
    df = preprocessor.feature_engineering(df)
    print(f"✓ Created advanced features")
    print(f"✓ Total features: {df.shape[1]}")
    
    # Display some engineered features
    engineered_features = [
        'bed_bath_ratio', 'facilities_score', 'luxury_index',
        'location_score', 'space_per_bedroom', 'is_premium'
    ]
    available = [f for f in engineered_features if f in df.columns]
    print(f"✓ Engineered features: {', '.join(available)}")
    
    # =========================================================================
    # STEP 6: Prepare Features (with LOG transformation)
    # =========================================================================
    print_subsection("Step 6: Preparing features for modeling")
    
    # Use log transformation for target (CRITICAL for price prediction)
    X_train, X_test, y_train, y_test, y_train_log, y_test_log = preprocessor.prepare_features(
        df,
        target_col='price',
        test_size=0.2,
        use_log_target=True
    )
    
    feature_names = preprocessor.feature_names
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Applied log transformation to target variable")
    
    # Get feature info
    feature_info = preprocessor.get_feature_info()
    print(f"\nFeature Breakdown:")
    print(f"  Binary features: {len(feature_info['binary_features'])}")
    print(f"  Continuous features: {len(feature_info['continuous_features'])}")
    
    # =========================================================================
    # STEP 7: Save Preprocessing Objects
    # =========================================================================
    print_subsection("Step 7: Saving preprocessing objects")
    
    os.makedirs('models/saved_models', exist_ok=True)
    
    import joblib
    joblib.dump(preprocessor.feature_scaler, 'models/saved_models/scaler.pkl')
    joblib.dump(feature_names, 'models/saved_models/feature_names.pkl')
    joblib.dump((X_train, X_test, y_train, y_test), 'models/saved_models/train_test_data.pkl')
    
    print("✓ Saved scaler")
    print("✓ Saved feature names")
    print("✓ Saved train/test data for explainability")
    
    # =========================================================================
    # STEP 8: Train All Models
    # =========================================================================
    print_section("TRAINING ALL MODELS WITH OPTIMIZED HYPERPARAMETERS")
    
    results = models.train_all_models(
        X_train, 
        y_train_log,  # Use log-transformed target
        X_test, 
        y_test_log,   # Use log-transformed test target
        use_log=True
    )
    
    # =========================================================================
    # STEP 9: Display Results
    # =========================================================================
    print_section("TRAINING RESULTS SUMMARY")
    
    # Create results DataFrame
    results_data = []
    for model_name, metrics in models.model_metrics.items():
        results_data.append({
            'Model': model_name,
            'R² Score': metrics.get('r2_score', 0),
            'RMSE': metrics.get('rmse', 0),
            'MAE': metrics.get('mae', 0),
            'MAPE': metrics.get('mape', 0)
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('R² Score', ascending=False)
    
    # Pretty print table
    print("┌" + "─" * 78 + "┐")
    print(f"│ {'Model':<20} │ {'R² Score':<10} │ {'RMSE':<15} │ {'MAE':<15} │")
    print("├" + "─" * 78 + "┤")
    
    for _, row in results_df.iterrows():
        print(f"│ {row['Model']:<20} │ {row['R² Score']:<10.4f} │ ₹{row['RMSE']:<14,.0f} │ ₹{row['MAE']:<14,.0f} │")
    
    print("└" + "─" * 78 + "┘")
    
    # Highlight best model
    best_model_display = models.get_best_model()
    best_metrics = models.model_metrics[best_model_display]
    
    print(f"\n{'='*80}")
    print(f"🏆 BEST MODEL: {best_model_display.upper()}")
    print(f"   R² Score: {best_metrics['r2_score']:.4f} (explains {best_metrics['r2_score']*100:.1f}% of variance)")
    print(f"   RMSE: ₹{best_metrics['rmse']:,.0f}")
    print(f"   MAE: ₹{best_metrics['mae']:,.0f}")
    print(f"   MAPE: {best_metrics['mape']:.2f}%")
    print(f"{'='*80}")
    
    # Performance comparison
    print("\n📊 Performance Analysis:")
    
    baseline_r2 = results_df.iloc[-1]['R² Score']
    best_r2 = results_df.iloc[0]['R² Score']
    improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100
    
    print(f"  Baseline (worst) R²: {baseline_r2:.4f}")
    print(f"  Best model R²: {best_r2:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Check if Neural Network is working
    nn_metrics = models.model_metrics.get('Neural Network')
    if nn_metrics:
        if nn_metrics['r2_score'] > 0:
            print(f"  ✓ Neural Network FIXED: R² = {nn_metrics['r2_score']:.4f}")
        else:
            print(f"  ⚠️  Neural Network needs tuning: R² = {nn_metrics['r2_score']:.4f}")
    
    # =========================================================================
    # STEP 10: Save Models
    # =========================================================================
    print_subsection("Step 10: Saving trained models")
    
    models.save_models('models/saved_models/')
    print(f"✓ Saved {len(models.models)} models")
    
    # =========================================================================
    # STEP 11: Save Training Results
    # =========================================================================
    print_subsection("Step 11: Saving training results")
    
    # Comprehensive results dictionary
    results_dict = {
        'training_date': datetime.now().isoformat(),
        'dataset': 'Housing.csv',
        'total_records': len(df),
        'training_records': len(X_train),
        'test_records': len(X_test),
        'features': feature_names,
        'feature_engineering': {
            'total_features': len(feature_names),
            'binary_features': len(feature_info['binary_features']),
            'continuous_features': len(feature_info['continuous_features'])
        },
        'preprocessing': {
            'outliers_removed': original_count - len(df),
            'target_transformation': 'log',
            'feature_scaling': 'RobustScaler (continuous only)'
        },
        'best_model': best_model_display,
        'models': {}
    }
    
    # Add model results
    for model_name, metrics in models.model_metrics.items():
        results_dict['models'][model_name] = {
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'r2_score': float(metrics['r2_score']),
            'mape': float(metrics['mape'])
        }
    
    # Save to JSON
    with open('models/training_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    
    print("✓ Saved training_results.json")
    
    # Save model state for Streamlit
    model_state = {
        'trained': True,
        'feature_names': feature_names,
        'best_model': best_model_display,
        'model_count': len(models.models),
        'training_date': datetime.now().isoformat(),
        'performance': {
            'best_r2': float(best_metrics['r2_score']),
            'best_rmse': float(best_metrics['rmse'])
        }
    }
    
    with open('models/model_state.json', 'w') as f:
        json.dump(model_state, f, indent=4)
    
    print("✓ Saved model_state.json")
    
    # =========================================================================
    # STEP 12: Sample Predictions
    # =========================================================================
    print_section("SAMPLE PREDICTIONS")
    
    # Get 5 random test samples
    indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for i, idx in enumerate(indices, 1):
        actual = y_test.iloc[idx]
        
        # Get prediction using best model
        predicted = models.predict(X_test.iloc[[idx]])[0]
        
        error = predicted - actual
        error_pct = (error / actual) * 100
        
        # Get original features
        original_idx = y_test.index[idx]
        
        print(f"Property {i}:")
        if 'area' in df.columns:
            print(f"  Area: {int(df.loc[original_idx, 'area'])} sq ft")
        if 'bedrooms' in df.columns:
            print(f"  Bedrooms: {int(df.loc[original_idx, 'bedrooms'])}")
        if 'bathrooms' in df.columns:
            print(f"  Bathrooms: {int(df.loc[original_idx, 'bathrooms'])}")
        
        print(f"  Actual Price:    ₹{actual:,.0f}")
        print(f"  Predicted Price: ₹{predicted:,.0f}")
        print(f"  Error:           ₹{error:,.0f} ({error_pct:+.1f}%)")
        print()
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("TRAINING COMPLETE!")
    
    print("📦 Files Saved:")
    print("   ✓ models/saved_models/ - All 8 models + preprocessing objects")
    print("   ✓ models/training_results.json - Comprehensive metrics")
    print("   ✓ models/model_state.json - Quick-load state")
    
    print(f"\n🏆 Best Model: {best_model_display}")
    print(f"   R² Score: {best_metrics['r2_score']:.4f}")
    print(f"   RMSE: ₹{best_metrics['rmse']:,.0f}")
    
    print("\n✨ Improvements Applied:")
    print("   ✓ Log transformation on target variable")
    print("   ✓ Outlier removal (IQR method)")
    print("   ✓ Advanced feature engineering (20+ features)")
    print("   ✓ Proper scaling (continuous features only)")
    print("   ✓ Optimized hyperparameters for all models")
    print("   ✓ Fixed Neural Network (proper target scaling)")
    
    print("\n📊 Performance Expectations:")
    if best_r2 > 0.80:
        print(f"   🎉 EXCELLENT! R² > 0.80 achieved!")
    elif best_r2 > 0.75:
        print(f"   👍 GOOD! R² > 0.75 achieved")
    else:
        print(f"   ⚠️  Consider further tuning (current R² = {best_r2:.4f})")
    
    print("\n🚀 Next Steps:")
    print("   1. Run: streamlit run app/streamlit_app.py")
    print("   2. Navigate to '📊 Model Insights' page")
    print("   3. View detailed model performance and explanations")
    
    print("\n" + "=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()