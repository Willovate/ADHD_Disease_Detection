"""
Test Script - View Model Accuracy and Metrics
"""
import os
import pickle
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))

print("\n" + "="*70)
print("ADHD DISEASE DETECTION - MODEL TEST & ACCURACY CHECK")
print("="*70 + "\n")

# Check if metrics file exists
metrics_file = os.path.join(base_dir, "model_metrics.txt")
if os.path.exists(metrics_file):
    print("📊 MODEL PERFORMANCE METRICS:\n")
    with open(metrics_file, 'r') as f:
        print(f.read())
else:
    print("❌ Metrics file not found. Please run 'python main.py' first to train the model.")
    print("\nTo train your model:")
    print("  1. Activate venv: .venv\\Scripts\\Activate.ps1")
    print("  2. Run: python main.py")

# Check what files have been generated
print("\n" + "="*70)
print("GENERATED FILES:")
print("="*70 + "\n")

files_to_check = [
    ("adhd_model.pkl", "Trained RandomForest Model"),
    ("scaler.pkl", "Data Scaler"),
    ("model_info.pkl", "Model Information"),
    ("confusion_matrix.png", "Confusion Matrix Visualization"),
    ("feature_importance.png", "Feature Importance Plot"),
    ("waveform_comparison.png", "Waveform Comparison Plot"),
    ("model_metrics.txt", "Performance Metrics (Text)")
]

for filename, description in files_to_check:
    filepath = os.path.join(base_dir, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.2f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.2f} KB"
        else:
            size_str = f"{size} Bytes"
        print(f"✓ {filename:30} - {description:40} ({size_str})")
    else:
        print(f"✗ {filename:30} - {description:40} (NOT FOUND)")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("\n1. View visualization files (.png) - Double-click to open")
print("2. Check model_metrics.txt for detailed accuracy")
print("3. To retrain or test on new data, run: python main.py")
print("\n" + "="*70 + "\n")
