# src/data_preprocessing.py - IMPROVED VERSION

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class RealEstateDataPreprocessor:
    """Enhanced data preprocessing for real estate with advanced feature engineering"""
    
    def __init__(self):
        self.feature_scaler = RobustScaler()  # Better for outliers than StandardScaler
        self.target_scaler = None
        self.feature_names = None
        self.continuous_features = []
        self.binary_features = []
        
    def load_data(self, file_path):
        """Load real estate data from CSV"""
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df)} records from {file_path}")
        return df
    
    def remove_outliers(self, df, columns=None):
        """Remove outliers using IQR method"""
        if columns is None:
            columns = ['price', 'area']
        
        df_clean = df.copy()
        original_count = len(df_clean)
        
        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
        
        removed = original_count - len(df_clean)
        if removed > 0:
            print(f"✓ Removed {removed} outliers ({removed/original_count*100:.1f}%)")
        
        return df_clean
    
    def clean_data(self, df):
        """Clean and handle missing values with intelligent strategies"""
        df_clean = df.copy()
        
        # Handle missing values in numerical columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use median for robustness to outliers
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Handle categorical missing values
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Remove duplicates
        original_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = original_count - len(df_clean)
        
        if duplicates_removed > 0:
            print(f"✓ Removed {duplicates_removed} duplicate records")
        
        return df_clean
    
    def process_housing_data(self, df):
        """Process Housing.csv with proper encoding"""
        df_processed = df.copy()
        
        # Convert yes/no columns to binary (0/1)
        binary_columns = [
            'mainroad', 'guestroom', 'basement', 
            'hotwaterheating', 'airconditioning', 'prefarea'
        ]
        
        for col in binary_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map({'yes': 1, 'no': 0})
        
        # One-hot encode furnishing status (better than ordinal for non-ordered categories)
        if 'furnishingstatus' in df_processed.columns:
            df_processed = pd.get_dummies(
                df_processed, 
                columns=['furnishingstatus'], 
                drop_first=True,  # Avoid multicollinearity
                prefix='furnish'
            )
        
        print("✓ Processed housing-specific features")
        return df_processed
    
    def feature_engineering(self, df):
        """Advanced feature engineering for real estate"""
        df_eng = df.copy()
        
        # Basic ratio features
        if 'price' in df_eng.columns and 'area' in df_eng.columns:
            df_eng['price_per_sqft'] = df_eng['price'] / df_eng['area']
        
        if 'bedrooms' in df_eng.columns and 'bathrooms' in df_eng.columns:
            df_eng['bed_bath_ratio'] = df_eng['bedrooms'] / (df_eng['bathrooms'] + 1)
            df_eng['total_rooms'] = df_eng['bedrooms'] + df_eng['bathrooms']
        
        if 'area' in df_eng.columns and 'bedrooms' in df_eng.columns:
            df_eng['space_per_bedroom'] = df_eng['area'] / (df_eng['bedrooms'] + 1)
            df_eng['room_density'] = (df_eng['bedrooms'] + df_eng.get('bathrooms', 0)) / df_eng['area']
        
        if 'bathrooms' in df_eng.columns and 'bedrooms' in df_eng.columns:
            df_eng['bathroom_adequacy'] = df_eng['bathrooms'] / (df_eng['bedrooms'] + 1)
        
        # Facility scores
        facility_cols = ['mainroad', 'guestroom', 'basement', 
                        'hotwaterheating', 'airconditioning']
        available_facility_cols = [col for col in facility_cols if col in df_eng.columns]
        
        if available_facility_cols:
            df_eng['facilities_score'] = df_eng[available_facility_cols].sum(axis=1)
            
            # Weighted luxury index
            weights = {
                'guestroom': 1.5,
                'basement': 1.2,
                'hotwaterheating': 0.8,
                'airconditioning': 1.0
            }
            
            luxury_score = 0
            for col, weight in weights.items():
                if col in df_eng.columns:
                    luxury_score += df_eng[col] * weight
            
            df_eng['luxury_index'] = luxury_score
        
        # Location quality score
        if 'mainroad' in df_eng.columns and 'prefarea' in df_eng.columns:
            df_eng['location_score'] = (
                df_eng['mainroad'] * 2.0 + 
                df_eng['prefarea'] * 3.0
            )
        
        # Binary indicators
        if 'stories' in df_eng.columns:
            df_eng['is_multistory'] = (df_eng['stories'] > 1).astype(int)
        
        if 'parking' in df_eng.columns:
            df_eng['has_parking'] = (df_eng['parking'] > 0).astype(int)
        
        # Premium property indicator
        if all(col in df_eng.columns for col in ['area', 'prefarea', 'facilities_score']):
            df_eng['is_premium'] = (
                (df_eng['area'] > df_eng['area'].quantile(0.75)) &
                (df_eng['prefarea'] == 1) &
                (df_eng['facilities_score'] >= 3)
            ).astype(int)
        
        # Polynomial features for area
        if 'area' in df_eng.columns:
            df_eng['area_squared'] = df_eng['area'] ** 2
            df_eng['area_log'] = np.log1p(df_eng['area'])
        
        # Interaction features
        if 'area' in df_eng.columns and 'bedrooms' in df_eng.columns:
            df_eng['area_x_bedrooms'] = df_eng['area'] * df_eng['bedrooms']
        
        if 'location_score' in df_eng.columns and 'luxury_index' in df_eng.columns:
            df_eng['location_x_luxury'] = df_eng['location_score'] * df_eng['luxury_index']
        
        # Area categories as one-hot (better than ordinal)
        if 'area' in df_eng.columns:
            df_eng['area_small'] = (df_eng['area'] < 3000).astype(int)
            df_eng['area_medium'] = ((df_eng['area'] >= 3000) & (df_eng['area'] < 6000)).astype(int)
            df_eng['area_large'] = (df_eng['area'] >= 6000).astype(int)
        
        print(f"✓ Feature engineering complete: {df_eng.shape[1]} total features")
        return df_eng
    
    def prepare_features(self, df, target_col='price', test_size=0.2, use_log_target=True):
        """
        Prepare features with proper scaling strategy
        
        Parameters:
        - use_log_target: If True, applies log transformation to target (recommended for prices)
        """
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Drop price_per_sqft if it exists (it leaks target information)
        features_to_drop = [target_col]
        if 'price_per_sqft' in df.columns:
            features_to_drop.append('price_per_sqft')
        
        X = df.drop(columns=features_to_drop)
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Identify feature types
        self.binary_features = [col for col in X.columns if X[col].nunique() <= 2]
        self.continuous_features = [col for col in X.columns if col not in self.binary_features]
        
        print(f"✓ Identified {len(self.binary_features)} binary and {len(self.continuous_features)} continuous features")
        
        # Scale only continuous features (binary features stay 0/1)
        X_scaled = X.copy()
        if self.continuous_features:
            X_scaled[self.continuous_features] = self.feature_scaler.fit_transform(
                X[self.continuous_features]
            )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Apply log transformation to target if requested
        if use_log_target:
            y_train_log = np.log1p(y_train)
            y_test_log = np.log1p(y_test)
            print("✓ Applied log transformation to target variable")
            
            return X_train, X_test, y_train, y_test, y_train_log, y_test_log
        else:
            return X_train, X_test, y_train, y_test
    
    def inverse_transform_target(self, y_log):
        """Convert log-transformed predictions back to original scale"""
        return np.expm1(y_log)
    
    def get_feature_info(self):
        """Return information about processed features"""
        return {
            'total_features': len(self.feature_names) if self.feature_names else 0,
            'binary_features': self.binary_features,
            'continuous_features': self.continuous_features,
            'feature_names': self.feature_names
        }


# Example usage and testing
if __name__ == "__main__":
    preprocessor = RealEstateDataPreprocessor()
    
    try:
        # Load Housing.csv
        df = preprocessor.load_data('Housing.csv')
        
        # Process housing data
        df = preprocessor.process_housing_data(df)
        
        # Remove outliers
        df = preprocessor.remove_outliers(df, columns=['price', 'area'])
        
        # Clean data
        df = preprocessor.clean_data(df)
        
        # Feature engineering
        df = preprocessor.feature_engineering(df)
        
        # Prepare for modeling with log transformation
        X_train, X_test, y_train, y_test, y_train_log, y_test_log = preprocessor.prepare_features(
            df, 
            target_col='price',
            use_log_target=True
        )
        
        print(f"\n✓ Data preparation complete!")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")
        print(f"  Features: {len(preprocessor.feature_names)}")
        
        # Display feature info
        info = preprocessor.get_feature_info()
        print(f"\nFeature Breakdown:")
        print(f"  Binary features: {len(info['binary_features'])}")
        print(f"  Continuous features: {len(info['continuous_features'])}")
        
    except FileNotFoundError:
        print("❌ Housing.csv not found in current directory")
    except Exception as e:
        print(f"❌ Error: {e}")