#!/usr/bin/env python3
import sys
packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'flask', 'requests', 'jupyter']
print("Testing imports...")
print("-" * 50)
all_good = True
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg:20s} OK")
    except ImportError:
        print(f"❌ {pkg:20s} FAILED")
        all_good = False
print("-" * 50)
if all_good:
    print("\n🎉 All packages installed successfully!")
    print(f"Python: {sys.version}")
else:
    print("\n⚠️ Some packages failed")
