# train.py - FINAL ACCURACY-BOOSTED VERSION

import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from src.predictive_models import RealEstatePredictiveModels

print("=" * 80)
print("HOUSING PRICE PREDICTION - FINAL TRAINING")
print("=" * 80)

# Load data
df = pd.read_csv("Housing.csv")

# Encode categorical
binary_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["furnishingstatus"] = df["furnishingstatus"].map({
    "furnished": 2, "semi-furnished": 1, "unfurnished": 0
})

# Feature engineering
df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
df["facilities_score"] = (
    df["mainroad"] + df["guestroom"] +
    df["basement"] + df["hotwaterheating"] +
    df["airconditioning"]
)

# 🔥 IMPORTANT IMPROVEMENT
df["log_area"] = np.log1p(df["area"])

# Clean
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Features & log target
X = df.drop(columns=["price"])
y = np.log1p(df["price"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
models = RealEstatePredictiveModels()
metrics = models.train_all_models(X_train, y_train, X_test, y_test)

models.save_models()

# Save results
results = {
    "training_date": datetime.now().isoformat(),
    "records": len(df),
    "features": list(X.columns),
    "best_model": models.best_model_name,
    "metrics": metrics
}

with open("models/training_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Sample predictions
print("\nSAMPLE PREDICTIONS")
for idx in np.random.choice(len(X_test), 5, replace=False):
    actual = np.expm1(y_test.iloc[idx])
    predicted = models.predict(X_test.iloc[[idx]])[0]
    error = (predicted - actual) / actual * 100

    print(f"\nActual Price   : ₹{actual:,.0f}")
    print(f"Predicted Price: ₹{predicted:,.0f}")
    print(f"Error          : {error:+.2f}%")

print("\n✅ TRAINING COMPLETE")
print(f"🏆 Best Model: {models.best_model_name}")
