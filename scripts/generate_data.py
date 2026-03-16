#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 1000

print("Generating synthetic CI/CD pipeline data...")

data = {
    'build_id': range(1, n + 1),
    'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 180)) for _ in range(n)],
    'project_name': np.random.choice(['microservice-a', 'web-app', 'api-gateway', 'mobile-app'], n),
    'repo_size_mb': np.random.randint(50, 5000, n),
    'num_files': np.random.randint(100, 15000, n),
    'num_dependencies': np.random.randint(20, 800, n),
    'test_count': np.random.randint(10, 500, n),
    'cpu_cores': np.random.choice([2, 4, 8, 16], n),
    'memory_gb': np.random.choice([4, 8, 16, 32], n),
    'build_duration_sec': np.random.randint(120, 3600, n),
    'status': np.random.choice(['SUCCESS', 'FAILURE'], n, p=[0.78, 0.22]),
    'cost_usd': np.random.uniform(0.15, 8.0, n)
}

df = pd.DataFrame(data)

# Add realistic correlations
df['build_duration_sec'] = df['build_duration_sec'] + (df['repo_size_mb'] / 20).astype(int)
df['build_duration_sec'] = df['build_duration_sec'] + (df['num_files'] / 50).astype(int)
df['cost_usd'] = (df['cpu_cores'] * 0.048 + df['memory_gb'] * 0.0104) * (df['build_duration_sec'] / 3600)

# Save
df.to_csv('data/pipeline_builds.csv', index=False)
print(f"✅ Generated {n} records")
print(f"✅ Saved to data/pipeline_builds.csv")
print(f"\nDataset Info:")
print(df.info())
print(f"\nSuccess Rate: {(df['status']=='SUCCESS').mean()*100:.1f}%")
print(f"Avg Build Time: {df['build_duration_sec'].mean():.0f} seconds")
print(f"Avg Cost: ${df['cost_usd'].mean():.2f}")
