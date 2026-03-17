#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 1000

print("Generating improved synthetic CI/CD pipeline data...")

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
    'status': np.random.choice(['SUCCESS', 'FAILURE'], n, p=[0.78, 0.22]),
}

df = pd.DataFrame(data)

# IMPROVED: Build time calculation with STRONG CPU/RAM correlation
base_time = 600  # 10 minutes base

# Large repos take longer
repo_factor = (df['repo_size_mb'] / 100).values

# More files = longer
file_factor = (df['num_files'] / 100).values

# More deps = longer
dep_factor = (df['num_dependencies'] / 10).values

# MORE CPU = FASTER (inverse relationship!)
cpu_factor = 16 / df['cpu_cores'].values  # 2 cores = 8x slower than 16

# MORE RAM = SLIGHTLY FASTER
ram_factor = 32 / df['memory_gb'].values  # Less RAM = slower

# Calculate build duration with strong correlations
df['build_duration_sec'] = (
    base_time * cpu_factor * ram_factor +
    repo_factor * 2 +
    file_factor * 1.5 +
    dep_factor * 0.5 +
    np.random.randint(-60, 60, n)  # Random variation
).astype(int)

# Ensure minimum time
df['build_duration_sec'] = df['build_duration_sec'].clip(lower=120)

# Cost calculation (CPU and RAM based)
df['cost_usd'] = (df['cpu_cores'] * 0.048 + df['memory_gb'] * 0.0104) * (df['build_duration_sec'] / 3600)

# Adjust failures based on resources
# Low resources = more failures
resource_score = df['cpu_cores'] / 16 + df['memory_gb'] / 32
df.loc[resource_score < 0.3, 'status'] = np.random.choice(
    ['SUCCESS', 'FAILURE'], 
    size=(resource_score < 0.3).sum(), 
    p=[0.6, 0.4]
)

# Save
df.to_csv('data/pipeline_builds.csv', index=False)
print(f"✅ Generated {n} records with strong CPU/RAM correlations")
print(f"✅ Saved to data/pipeline_builds.csv")

# Show examples
print("\nExample data (different configs):")
for cpu in [2, 4, 8, 16]:
    for ram in [4, 8, 16, 32]:
        sample = df[(df['cpu_cores'] == cpu) & (df['memory_gb'] == ram)].head(1)
        if len(sample) > 0:
            print(f"{cpu} cores, {ram}GB RAM → {sample['build_duration_sec'].values[0]/60:.1f} min, ${sample['cost_usd'].values[0]:.3f}")
            break
