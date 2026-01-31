# src/model_training.py

"""
Complete model training pipeline
Trains all models and saves them for production use
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

from src.data_preprocessing import RealEstateDataPreprocessor
from src.predictive_models import RealEstatePredictiveModels

class ModelTrainingPipeline:
    """Complete pipeline for training and evaluating models"""
    
    def __init__(self, data_path='data/sample_data.csv'):
        self.data_path = data_path
        self.preprocessor = RealEstateDataPreprocessor()
        self.models = RealEstatePredictiveModels()
        self.results = {}
        
    def run_pipeline(self):
        """Execute complete training pipeline"""
        
        print("=" * 70)
        print("REAL ESTATE PRICE PREDICTION - MODEL TRAINING PIPELINE")
        print("=" * 70)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Step 1: Load Data
        print("Step 1: Loading data...")
        if not os.path.exists(self.data_path):
            print(f"Data file not found. Creating sample dataset...")
            df = self.preprocessor.create_sample_dataset(n_samples=1000)
            os.makedirs('data', exist_ok=True)
            df.to_csv(self.data_path, index=False)
            print(f"✓ Sample dataset created: {self.data_path}")
        else:
            df = self.preprocessor.load_data(self.data_path)
            print(f"✓ Data loaded: {len(df)} records")
        
        # Step 2: Data Cleaning
        print("\nStep 2: Cleaning data...")
        df = self.preprocessor.clean_data(df)
        print(f"✓ Data cleaned: {len(df)} records remaining")
        
        # Step 3: Feature Engineering
        print("\nStep 3: Feature engineering...")
        df = self.preprocessor.feature_engineering(df)
        print(f"✓ Features engineered: {df.shape[1]} total features")
        
        # Step 4: Encode Categorical Variables
        print("\nStep 4: Encoding categorical variables...")
        categorical_cols = ['location', 'property_type']
        df_encoded = self.preprocessor.encode_categorical(df, categorical_cols)
        print(f"✓ Categorical variables encoded")
        
        # Step 5: Prepare Train/Test Split
        print("\nStep 5: Splitting data...")
        X_train, X_test, y_train, y_test = self.preprocessor.prepare_features(df_encoded)
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        # Step 6: Train All Models
        print("\n" + "=" * 70)
        print("Step 6: Training models...")
        print("=" * 70 + "\n")
        
        self.results = self.models.train_all_models(X_train, y_train, X_test, y_test)
        
        # Step 7: Display Results
        print("\n" + "=" * 70)
        print("TRAINING RESULTS SUMMARY")
        print("=" * 70 + "\n")
        
        self._display_results()
        
        # Step 8: Save Models
        print("\n" + "=" * 70)
        print("Step 7: Saving models...")
        print("=" * 70 + "\n")
        
        os.makedirs('models/saved_models', exist_ok=True)
        self.models.save_models('models/saved_models/')
        print("✓ All models saved successfully")
        
        # Step 9: Save Results
        self._save_results()
        
        print("\n" + "=" * 70)
        print(f"PIPELINE COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        return self.results
    
    def _display_results(self):
        """Display formatted results"""
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R² Score': metrics['r2_score'],
                'MAPE (%)': metrics['mape']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {self.models.best_model_name.upper()}")
        print(f"R² Score: {self.results[self.models.best_model_name]['r2_score']:.4f}")
        print(f"RMSE: ${self.results[self.models.best_model_name]['rmse']:,.2f}")
        print(f"{'='*70}")
    
    def _save_results(self):
        """Save training results to JSON"""
        
        results_dict = {}
        for model_name, metrics in self.results.items():
            results_dict[model_name] = {
                'mse': float(metrics['mse']),
                'rmse': float(metrics['rmse']),
                'mae': float(metrics['mae']),
                'r2_score': float(metrics['r2_score']),
                'mape': float(metrics['mape'])
            }
        
        results_dict['best_model'] = self.models.best_model_name
        results_dict['timestamp'] = datetime.now().isoformat()
        
        os.makedirs('models', exist_ok=True)
        with open('models/training_results.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print("\n✓ Results saved to: models/training_results.json")
    
    def generate_prediction_report(self, n_samples=5):
        """Generate sample predictions report"""
        
        print("\n" + "=" * 70)
        print("SAMPLE PREDICTIONS")
        print("=" * 70 + "\n")
        
        # Load data for predictions
        df = pd.read_csv(self.data_path)
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.feature_engineering(df)
        
        categorical_cols = ['location', 'property_type']
        df_encoded = self.preprocessor.encode_categorical(df, categorical_cols)
        
        X = df_encoded.drop(columns=['price'])
        y = df_encoded['price']
        
        # Scale features
        X_scaled = self.preprocessor.scaler.transform(X)
        
        # Get random samples
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        for idx in indices:
            actual_price = y.iloc[idx]
            predicted_price = self.models.predict(X_scaled[idx:idx+1])[0]
            error = predicted_price - actual_price
            error_pct = (error / actual_price) * 100
            
            print(f"Property {idx + 1}:")
            print(f"  Actual Price:    ${actual_price:,.2f}")
            print(f"  Predicted Price: ${predicted_price:,.2f}")
            print(f"  Error:           ${error:,.2f} ({error_pct:+.2f}%)")
            print()

def main():
    """Main execution function"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train real estate prediction models')
    parser.add_argument('--data', type=str, default='data/sample_data.csv',
                       help='Path to training data CSV')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to generate if creating new dataset')
    parser.add_argument('--report', action='store_true',
                       help='Generate prediction report after training')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = ModelTrainingPipeline(data_path=args.data)
    results = pipeline.run_pipeline()
    
    # Generate report if requested
    if args.report:
        pipeline.generate_prediction_report()
    
    return results

if __name__ == "__main__":
    main()