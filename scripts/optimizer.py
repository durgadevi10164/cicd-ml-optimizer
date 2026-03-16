#!/usr/bin/env python3
import joblib
import pandas as pd
import numpy as np

class PipelineOptimizer:
    def __init__(self):
        self.duration_model = joblib.load('models/duration_predictor.pkl')
        self.success_model = joblib.load('models/success_predictor.pkl')
    
    def optimize_resources(self, repo_size, num_files, num_deps, test_count):
        """Find optimal CPU/Memory configuration"""
        configs = []
        cpu_options = [2, 4, 8, 16]
        mem_options = [4, 8, 16, 32]
        
        for cpu in cpu_options:
            for mem in mem_options:
                X = pd.DataFrame([{
                    'repo_size_mb': repo_size,
                    'num_files': num_files,
                    'num_dependencies': num_deps,
                    'test_count': test_count,
                    'cpu_cores': cpu,
                    'memory_gb': mem
                }])
                
                duration = self.duration_model.predict(X)[0]
                success_prob = self.success_model.predict_proba(X)[0][1]
                cost = (cpu * 0.048 + mem * 0.0104) * (duration / 3600)
                score = cost / success_prob
                
                configs.append({
                    'cpu_cores': cpu,
                    'memory_gb': mem,
                    'predicted_duration_sec': round(duration, 2),
                    'predicted_duration_min': round(duration/60, 2),
                    'success_probability': round(success_prob * 100, 2),
                    'estimated_cost_usd': round(cost, 3),
                    'score': round(score, 4)
                })
        
        configs_df = pd.DataFrame(configs).sort_values('score')
        
        return {
            'recommended': configs_df.iloc[0].to_dict(),
            'all_options': configs_df.to_dict('records')
        }

if __name__ == "__main__":
    print("Testing Pipeline Optimizer...")
    print("=" * 70)
    
    optimizer = PipelineOptimizer()
    
    result = optimizer.optimize_resources(
        repo_size=2000,
        num_files=5000,
        num_deps=300,
        test_count=150
    )
    
    print("\n🎯 RECOMMENDED CONFIGURATION:")
    rec = result['recommended']
    print(f"CPU: {rec['cpu_cores']} cores")
    print(f"Memory: {rec['memory_gb']} GB")
    print(f"Expected Duration: {rec['predicted_duration_min']} minutes")
    print(f"Success Probability: {rec['success_probability']}%")
    print(f"Estimated Cost: ${rec['estimated_cost_usd']}")
