#!/usr/bin/env python3
"""
Diagnostic Script for Real Estate AI App
Checks for all required files and provides setup instructions
"""

import os
from pathlib import Path
import sys

def check_directory_structure():
    """Check if the required directory structure exists"""
    
    print("=" * 70)
    print("REAL ESTATE AI APP - DIAGNOSTIC TOOL")
    print("=" * 70)
    print()
    
    # Define expected structure
    required_structure = {
        'models/': {
            'saved_models/': {
                'files': [
                    'gradient_boosting.pkl',
                    'random_forest.pkl',
                    'xgboost.pkl',
                    'neural_network.pkl',
                    'ensemble.pkl',
                    'X_train.pkl',
                    'X_test.pkl',
                    'y_train.pkl',
                    'y_test.pkl',
                    'feature_names.pkl'
                ]
            },
            'files': [
                'model_state.json',
                'training_results.json'
            ]
        },
        'src/': {
            'files': [
                'data_preprocessing.py',
                'predictive_models.py',
                'investment_analytics.py',
                'explainability.py',
                'chatbot.py'
            ]
        },
        'app/': {
            'files': [
                'streamlit_app.py'
            ]
        },
        'data/': {
            'files': [
                'Housing.csv'
            ]
        }
    }
    
    # Try multiple possible root directories
    possible_roots = [
        Path.cwd(),
        Path(__file__).parent,
        Path(__file__).parent.parent,
    ]
    
    project_root = None
    for root in possible_roots:
        if (root / 'models').exists() or (root / 'src').exists() or (root / 'app').exists():
            project_root = root
            break
    
    if project_root is None:
        project_root = Path.cwd()
    
    print(f"📂 Checking directory: {project_root}")
    print()
    
    missing_items = []
    found_items = []
    
    def check_structure(base_path, structure, prefix=""):
        """Recursively check directory structure"""
        for item, content in structure.items():
            if item.endswith('/'):
                # It's a directory
                dir_path = base_path / item.rstrip('/')
                if dir_path.exists():
                    print(f"  ✅ {prefix}{item}")
                    found_items.append(f"{prefix}{item}")
                    # Check contents
                    if isinstance(content, dict):
                        check_structure(dir_path, content, prefix + "  ")
                else:
                    print(f"  ❌ {prefix}{item} - MISSING")
                    missing_items.append(f"{prefix}{item}")
            elif item == 'files':
                # Check files in this directory
                for file in content:
                    file_path = base_path / file
                    if file_path.exists():
                        size = file_path.stat().st_size
                        print(f"  ✅ {prefix}{file} ({size:,} bytes)")
                        found_items.append(f"{prefix}{file}")
                    else:
                        print(f"  ❌ {prefix}{file} - MISSING")
                        missing_items.append(f"{prefix}{file}")
    
    check_structure(project_root, required_structure)
    
    print()
    print("=" * 70)
    print(f"Summary: {len(found_items)} found, {len(missing_items)} missing")
    print("=" * 70)
    print()
    
    if missing_items:
        print("⚠️  MISSING COMPONENTS:")
        print()
        print("To set up the Real Estate AI App, you need to:")
        print()
        
        if any('models/' in item for item in missing_items):
            print("1️⃣  TRAIN THE MODELS:")
            print("   Run the training script to generate model files:")
            print("   ```")
            print("   python src/train_all.py")
            print("   ```")
            print()
        
        if any('src/' in item or 'data_preprocessing' in item for item in missing_items):
            print("2️⃣  CREATE SOURCE FILES:")
            print("   You need the following Python modules in the src/ directory:")
            print("   - data_preprocessing.py")
            print("   - predictive_models.py")
            print("   - investment_analytics.py")
            print("   - explainability.py")
            print("   - chatbot.py (optional)")
            print()
        
        if any('data/' in item for item in missing_items):
            print("3️⃣  PREPARE DATA:")
            print("   Place your real estate dataset in:")
            print("   data/real_estate_data.csv")
            print()
        
        print("📝 Detailed Missing Items:")
        for item in missing_items:
            print(f"   - {item}")
        print()
    else:
        print("✅ All required files found!")
        print()
        print("You can now run the Streamlit app:")
        print("```")
        print("streamlit run app/streamlit_app.py")
        print("```")
    
    return len(missing_items) == 0

def create_directory_structure():
    """Create the required directory structure"""
    
    print()
    response = input("Would you like to create the missing directories? (y/n): ")
    
    if response.lower() == 'y':
        dirs_to_create = [
            'models/saved_models',
            'src',
            'app',
            'data'
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")
        
        print()
        print("✅ Directory structure created!")
        print()
        print("Next steps:")
        print("1. Add your source files to src/")
        print("2. Add your data file to data/")
        print("3. Run: python src/train_all.py")
        print("4. Run: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    all_found = check_directory_structure()
    
    if not all_found:
        create_directory_structure()