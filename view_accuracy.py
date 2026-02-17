"""
Quick Accuracy Summary - View Your Model Performance
"""
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

print("\n" + "="*70)
print("YOUR ADHD DISEASE DETECTION MODEL - ACCURACY SUMMARY")
print("="*70 + "\n")

print("MODEL PERFORMANCE METRICS:")
print("-" * 70)
print("""
Your trained RandomForest model includes:
  - Dataset Size: 2,166,383 samples
  - Training Samples: 1,733,106 (80%)
  - Test Samples: 433,277 (20%)
  - Features: 19 EEG channels
  - Classes: ADHD vs Control (binary classification)
  - Model Type: RandomForest Classifier (100 trees)

VISUALIZATION FILES GENERATED:
""")

# List visualization files
print("\n1. CONFUSION MATRIX (confusion_matrix.png)")
print("   - Shows True Positives, True Negatives, False Positives, False Negatives")
print("   - Use to understand model classification performance")
print("   - Diagonal values show correct predictions\n")

print("2. FEATURE IMPORTANCE (feature_importance.png)")
print("   - Shows which EEG channels are most important for predictions")
print("   - Taller bars = more important features for ADHD detection\n")

print("3. WAVEFORM COMPARISON (waveform_comparison.png)")
print("   - Shows actual vs predicted classifications for 20 test samples")
print("   - Blue line = Actual labels")
print("   - Red line = Model predictions\n")

print("-" * 70)
print("\nTO VIEW YOUR RESULTS:")
print("  1. Open confusion_matrix.png to see model accuracy")
print("  2. Open feature_importance.png to see important EEG channels")
print("  3. Open waveform_comparison.png to see prediction examples\n")

print("TO GET DETAILED METRICS:")
print("  Run: .venv\\Scripts\\python.exe main.py")
print("  This will generate model_metrics.txt with accuracy percentage\n")

print("="*70 + "\n")

# Check if files exist
files = [
    "confusion_matrix.png",
    "feature_importance.png", 
    "waveform_comparison.png"
]

print("FILES AVAILABLE:")
for f in files:
    path = os.path.join(base_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024
        print(f"  ✓ {f:30} ({size:.1f} KB)")
    else:
        print(f"  ✗ {f:30} (NOT FOUND)")
        
print("\n" + "="*70 + "\n")
