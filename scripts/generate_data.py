#!/usr/bin/env python3
"""
Synthetic CI/CD Build Data Generator
Simulates historical build metrics for training
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

np.random.seed(42)

def generate_builds(n_builds=5000):
    """Generate synthetic CI/CD build data"""
    
    print(f"Generating {n_builds} synthetic builds...")
    
    builds = []
    start_date = datetime.now() - timedelta(days=180)
    
    for i in range(n_builds):
        # Time features
        timestamp = start_date + timedelta(
            hours=np.random.randint(0, 180*24)
        )
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Build characteristics
        repo_size_mb = np.random.lognormal(7, 1.5)  # Log-normal distribution
        num_files = int(np.random.lognormal(6, 1))
        test_count = int(np.random.lognormal(5, 1.2))
        dependencies = int(np.random.gamma(15, 3))
        
        # Resource allocation (what we'll optimize)
        cpu_cores = np.random.choice([2, 4, 8, 16], p=[0.4, 0.3, 0.2, 0.1])
        memory_gb = np.random.choice([4, 8, 16, 32], p=[0.35, 0.35, 0.2, 0.1])
        
        # Cache hit rate (affects build time)
        cache_hit_rate = np.random.beta(5, 2)
        
        # Branch type affects complexity
        branch_type = np.random.choice(
            ['main', 'feature', 'hotfix', 'release'],
            p=[0.2, 0.6, 0.1, 0.1]
        )
        branch_complexity = {
            'main': 1.2, 'feature': 1.0, 
            'hotfix': 0.8, 'release': 1.3
        }[branch_type]
        
        # Build time calculation (target variable)
        base_time = (
            repo_size_mb * 0.1 +
            num_files * 0.05 +
            test_count * 0.8 +
            dependencies * 0.3
        )
        
        # Resource efficiency
        resource_factor = (cpu_cores / 4) * (memory_gb / 8)
        time_reduction = 1 / (1 + 0.3 * resource_factor)
        
        # Cache benefit
        cache_benefit = 1 - (cache_hit_rate * 0.4)
        
        # Peak hours congestion
        congestion = 1.3 if 9 <= hour <= 17 else 1.0
        
        build_time_minutes = (
            base_time * 
            time_reduction * 
            cache_benefit * 
            branch_complexity * 
            congestion *
            np.random.normal(1, 0.1)  # Add noise
        )
        
        # Success/failure (worse with underprovisioned resources)
        failure_prob = 0.05
        if memory_gb < 8 and repo_size_mb > 1000:
            failure_prob = 0.25
        if cpu_cores == 2 and test_count > 500:
            failure_prob = 0.20
            
        status = 'failed' if np.random.random() < failure_prob else 'success'
        
        # Cost calculation ($/hour rates)
        cpu_cost_per_hour = {2: 0.05, 4: 0.10, 8: 0.20, 16: 0.40}[cpu_cores]
        mem_cost_per_hour = {4: 0.02, 8: 0.04, 16: 0.08, 32: 0.16}[memory_gb]
        
        cost_usd = (
            (cpu_cost_per_hour + mem_cost_per_hour) * 
            (build_time_minutes / 60)
        )
        
        builds.append({
            'build_id': f'build_{i:05d}',
            'timestamp': timestamp.isoformat(),
            'hour': hour,
            'day_of_week': day_of_week,
            'repo_size_mb': round(repo_size_mb, 2),
            'num_files': num_files,
            'test_count': test_count,
            'dependencies': dependencies,
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb,
            'cache_hit_rate': round(cache_hit_rate, 3),
            'branch_type': branch_type,
            'build_time_minutes': round(max(1, build_time_minutes), 2),
            'status': status,
            'cost_usd': round(cost_usd, 4)
        })
    
    df = pd.DataFrame(builds)
    print(f"\n✅ Generated {len(df)} builds")
    print(f"   Success rate: {(df['status']=='success').mean():.1%}")
    print(f"   Avg build time: {df['build_time_minutes'].mean():.1f} min")
    print(f"   Avg cost: ${df['cost_usd'].mean():.3f}")
    
    return df

if __name__ == '__main__':
    # Generate data
    df = generate_builds(5000)
    
    # Save
    output_path = 'data/builds.csv'
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    
    # Quick stats
    print("\n📊 Dataset Summary:")
    print(df.describe())
