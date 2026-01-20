#!/usr/bin/env python3
"""
Build Time Prediction & Resource Optimization
"""
import joblib
import numpy as np
import pandas as pd
import json

def load_model():
    """Load trained model and artifacts"""
    model = joblib.load('models/rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    le_branch = joblib.load('models/label_encoder.joblib')
    
    with open('models/metadata.json') as f:
        metadata = json.load(f)
    
    return model, scaler, le_branch, metadata

def predict_build_time(build_config):
    """Predict build time for given configuration"""
    model, scaler, le_branch, metadata = load_model()
    
    # Encode branch
    branch_encoded = le_branch.transform([build_config['branch_type']])[0]
    
    # Calculate derived features
    resource_score = build_config['cpu_cores'] * build_config['memory_gb']
    complexity_score = (
        build_config['repo_size_mb'] * 0.001 +
        build_config['num_files'] * 0.01 +
        build_config['test_count'] * 0.1
    )
    
    # Create feature vector
    features = np.array([[
        build_config['repo_size_mb'],
        build_config['num_files'],
        build_config['test_count'],
        build_config['dependencies'],
        build_config['cpu_cores'],
        build_config['memory_gb'],
        build_config.get('cache_hit_rate', 0.5),
        build_config.get('hour', 12),
        build_config.get('day_of_week', 2),
        branch_encoded,
        resource_score,
        complexity_score
    ]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    predicted_time = model.predict(features_scaled)[0]
    
    return max(1, predicted_time)

def optimize_resources(build_config):
    """Find optimal CPU/Memory configuration"""
    print("\n🔍 Optimizing resources...\n")
    
    cpu_options = [2, 4, 8, 16]
    mem_options = [4, 8, 16, 32]
    
    results = []
    
    for cpu in cpu_options:
        for mem in mem_options:
            config = build_config.copy()
            config['cpu_cores'] = cpu
            config['memory_gb'] = mem
            
            pred_time = predict_build_time(config)
            
            # Cost calculation
            cpu_cost_hr = {2: 0.05, 4: 0.10, 8: 0.20, 16: 0.40}[cpu]
            mem_cost_hr = {4: 0.02, 8: 0.04, 16: 0.08, 32: 0.16}[mem]
            cost = (cpu_cost_hr + mem_cost_hr) * (pred_time / 60)
            
            results.append({
                'cpu_cores': cpu,
                'memory_gb': mem,
                'predicted_time_min': round(pred_time, 2),
                'cost_usd': round(cost, 4),
                'cost_per_min': round(cost / pred_time, 4)
            })
    
    df_results = pd.DataFrame(results).sort_values('cost_usd')
    
    print("💰 Most Cost-Efficient:")
    print(df_results.head(3).to_string(index=False))
    
    print("\n⚡ Fastest Build Time:")
    fastest = df_results.nsmallest(3, 'predicted_time_min')
    print(fastest.to_string(index=False))
    
    # Recommended config (balance of speed and cost)
    recommended = df_results.iloc[0]
    
    print(f"\n✅ RECOMMENDED CONFIGURATION:")
    print(f"   CPU: {recommended['cpu_cores']} cores")
    print(f"   Memory: {recommended['memory_gb']} GB")
    print(f"   Estimated Time: {recommended['predicted_time_min']:.1f} min")
    print(f"   Estimated Cost: ${recommended['cost_usd']:.4f}")
    
    return recommended

if __name__ == '__main__':
    # Example build configuration
    example_build = {
        'repo_size_mb': 850,
        'num_files': 450,
        'test_count': 320,
        'dependencies': 45,
        'branch_type': 'feature',
        'cache_hit_rate': 0.6,
        'hour': 14,
        'day_of_week': 2,
        'cpu_cores': 4,  # Current config
        'memory_gb': 8   # Current config
    }
    
    print("📦 Build Configuration:")
    print(f"   Repo size: {example_build['repo_size_mb']} MB")
    print(f"   Files: {example_build['num_files']}")
    print(f"   Tests: {example_build['test_count']}")
    print(f"   Branch: {example_build['branch_type']}")
    
    # Predict with current config
    current_time = predict_build_time(example_build)
    print(f"\n⏱️  Current Config Prediction: {current_time:.1f} minutes")
    
    # Optimize
    optimize_resources(example_build)
