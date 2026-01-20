#!/usr/bin/env python3
"""
ML Model Training for CI/CD Optimization
Predicts build time and recommends optimal resources
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

def load_and_prepare_data(filepath='data/builds.csv'):
    """Load and preprocess build data"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Encode categorical
    le_branch = LabelEncoder()
    df['branch_encoded'] = le_branch.fit_transform(df['branch_type'])
    
    # Feature engineering
    df['resource_score'] = df['cpu_cores'] * df['memory_gb']
    df['complexity_score'] = (
        df['repo_size_mb'] * 0.001 + 
        df['num_files'] * 0.01 + 
        df['test_count'] * 0.1
    )
    
    # Features for training
    feature_cols = [
        'repo_size_mb', 'num_files', 'test_count', 'dependencies',
        'cpu_cores', 'memory_gb', 'cache_hit_rate',
        'hour', 'day_of_week', 'branch_encoded',
        'resource_score', 'complexity_score'
    ]
    
    X = df[feature_cols]
    y = df['build_time_minutes']
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"   Features: {len(feature_cols)}")
    
    return X, y, le_branch, feature_cols

def train_model(X, y):
    """Train Random Forest model"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n🤖 Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    metrics = {
        'train_mae': round(train_mae, 2),
        'test_mae': round(test_mae, 2),
        'train_r2': round(train_r2, 3),
        'test_r2': round(test_r2, 3)
    }
    
    print(f"\n✅ Training MAE: {train_mae:.2f} minutes")
    print(f"✅ Test MAE: {test_mae:.2f} minutes")
    print(f"✅ Test R²: {test_r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 5 Important Features:")
    print(feature_importance.head())
    
    return model, scaler, metrics, feature_importance

def save_artifacts(model, scaler, le_branch, metrics, feature_cols, feature_importance):
    """Save trained model and metadata"""
    print("\n💾 Saving model artifacts...")
    
    # Save model
    joblib.dump(model, 'models/rf_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(le_branch, 'models/label_encoder.joblib')
    
    # Save metadata
    metadata = {
        'metrics': metrics,
        'feature_columns': feature_cols,
        'feature_importance': feature_importance.to_dict('records')[:10],
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Saved:")
    print("   - models/rf_model.joblib")
    print("   - models/scaler.joblib")
    print("   - models/label_encoder.joblib")
    print("   - models/metadata.json")

if __name__ == '__main__':
    # Load data
    X, y, le_branch, feature_cols = load_and_prepare_data()
    
    # Train
    model, scaler, metrics, feature_importance = train_model(X, y)
    
    # Save
    save_artifacts(model, scaler, le_branch, metrics, feature_cols, feature_importance)
    
    print("\n🎉 Training complete!")
