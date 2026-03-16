#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import joblib

print("Training ML Models...")
print("=" * 70)

# Load data
df = pd.read_csv('data/pipeline_builds.csv')
print(f"Loaded {len(df)} build records\n")

# Features
features = ['repo_size_mb', 'num_files', 'num_dependencies', 'test_count', 'cpu_cores', 'memory_gb']

# ==== MODEL 1: Build Duration Prediction ====
print("Training Model 1: Build Duration Predictor")
print("-" * 70)

X = df[features]
y = df['build_duration_sec']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

duration_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
duration_model.fit(X_train, y_train)

y_pred = duration_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f} seconds ({mae/60:.2f} minutes)")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}% accuracy)")

joblib.dump(duration_model, 'models/duration_predictor.pkl')
print("✅ Saved to models/duration_predictor.pkl\n")

# ==== MODEL 2: Build Success Prediction ====
print("Training Model 2: Build Success Predictor")
print("-" * 70)

X_class = df[features]
y_class = df['status']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

success_model = RandomForestClassifier(n_estimators=100, random_state=42)
success_model.fit(X_train_c, y_train_c)

y_pred_c = success_model.predict(X_test_c)
accuracy = accuracy_score(y_test_c, y_pred_c)

print(f"Accuracy: {accuracy*100:.2f}%")

joblib.dump(success_model, 'models/success_predictor.pkl')
print("✅ Saved to models/success_predictor.pkl\n")

print("=" * 70)
print("🎉 All models trained successfully!")
print(f"Duration Model: {r2*100:.1f}% accurate")
print(f"Success Model: {accuracy*100:.1f}% accurate")
