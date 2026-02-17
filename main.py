# ADHD Disease Detection - Ensemble Learning Model
# Advanced Hyperparameter Tuning with XGBoost, Gradient Boosting & Random Forest

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to use 'Agg' backend
matplotlib.use('Agg')

# Get absolute path to dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "Dataset", "adhdata.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

print(f"Loading dataset from {file_path}")
df = pd.read_csv(file_path, low_memory=False)

print("\n--- Dataset Info ---")
print(f"Total Samples: {len(df):,}")
print(f"Columns: {df.shape[1]}")

# Handle Missing Values
df.dropna(inplace=True)

# Identify Target Column
target_column = "Class"
if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found")

# Encode Target Variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target_column].astype(str))
print(f"\nClass Distribution:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {u} ({label_encoder.classes_[u]}): {c:,} samples ({c/len(y)*100:.1f}%)")

# Select Features
feature_candidates = [col for col in df.columns if col not in {target_column, "ID"}]
X = df[feature_candidates].select_dtypes(include=[np.number])

print(f"\nSelected {X.shape[1]} numeric features")

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (with stratification)
# Sample data for faster training on large dataset
sample_size = min(500000, len(X_scaled))  # Use max 500K samples for speed
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_scaled_sample = X_scaled[sample_indices]
y_sample = y[sample_indices]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
)

print(f"\nTrain Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# ============================================================================
# 🎯 **MODEL 1: OPTIMIZED XGBoost (GPU-Accelerated)**
# ============================================================================
print("\n" + "="*70)
print("🚀 TRAINING OPTIMIZED XGBOOST MODEL")
print("="*70)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
scale_pos_weight = class_weights[1] / class_weights[0]

clf_xgb = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    tree_method='hist',
    n_jobs=-1,
    eval_metric='logloss',
    random_state=42,
    verbose=0
)

print("[OK] Building optimized XGBoost classifier...")
clf_xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("[OK] XGBoost training completed")

y_pred_xgb = clf_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\n✓ XGBoost Accuracy: {accuracy_xgb:.4f} ({accuracy_xgb*100:.2f}%)")

# ============================================================================
# 🎯 **MODEL 2: Gradient Boosting**
# ============================================================================
print("\n" + "="*70)
print("⚡ TRAINING GRADIENT BOOSTING MODEL")
print("="*70)

clf_gb = GradientBoostingClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.7,
    random_state=42,
    verbose=1
)

print("[OK] Building Gradient Boosting classifier...")
clf_gb.fit(X_train, y_train)
print("[OK] Gradient Boosting training completed")

y_pred_gb = clf_gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"\n✓ Gradient Boosting Accuracy: {accuracy_gb:.4f} ({accuracy_gb*100:.2f}%)")

# ============================================================================
# 🎯 **MODEL 3: Random Forest with Class Weights**
# ============================================================================
print("\n" + "="*70)
print("🌲 TRAINING OPTIMIZED RANDOM FOREST MODEL")
print("="*70)

clf_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=15,
    min_samples_leaf=8,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("[OK] Building Random Forest classifier...")
clf_rf.fit(X_train, y_train)
print("[OK] Random Forest training completed")

y_pred_rf = clf_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\n✓ Random Forest Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")

# ============================================================================
# 🎯 **ENSEMBLE: Voting Classifier**
# ============================================================================
print("\n" + "="*70)
print("🎪 ENSEMBLE VOTING (XGBoost + GradBoost + RandomForest)")
print("="*70)

# Soft voting using probabilities
y_pred_xgb_prob = clf_xgb.predict_proba(X_test)[:, 1]
y_pred_gb_prob = clf_gb.predict_proba(X_test)[:, 1]
y_pred_rf_prob = clf_rf.predict_proba(X_test)[:, 1]

# Average the probabilities (weights: 0.4, 0.3, 0.3)
y_pred_ensemble_prob = (0.4 * y_pred_xgb_prob + 0.3 * y_pred_gb_prob + 0.3 * y_pred_rf_prob)
y_pred_ensemble = (y_pred_ensemble_prob > 0.5).astype(int)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"\n✓ Ensemble Accuracy: {accuracy_ensemble:.4f} ({accuracy_ensemble*100:.2f}%)")

# ============================================================================
# 📊 **EVALUATION & COMPARISON**
# ============================================================================
print("\n" + "="*70)
print("📊 MODEL COMPARISON")
print("="*70)

baseline = 0.74
print(f"\n{'Model':<25} {'Accuracy':<15} {'Improvement':<15}")
print("-" * 55)
print(f"{'Baseline (74%)':<25} {baseline:<15.2%} {0.0:<15.2%}")
print(f"{'Optimized XGBoost':<25} {accuracy_xgb:<15.2%} {accuracy_xgb-baseline:<15.2%}")
print(f"{'Gradient Boosting':<25} {accuracy_gb:<15.2%} {accuracy_gb-baseline:<15.2%}")
print(f"{'Random Forest':<25} {accuracy_rf:<15.2%} {accuracy_rf-baseline:<15.2%}")
print(f"{'Ensemble (BEST)':<25} {accuracy_ensemble:<15.2%} {accuracy_ensemble-baseline:<15.2%}")
print("-" * 55)

# Use ensemble as final predictions
y_pred_final = y_pred_ensemble

# ============================================================================
# 💾 **SAVE ALL MODELS**
# ============================================================================
print("\n💾 Saving models...")

xgb_model_file = os.path.join(base_dir, "model_xgboost.pkl")
with open(xgb_model_file, 'wb') as f:
    pickle.dump(clf_xgb, f)
print(f"[OK] XGBoost model saved")

gb_model_file = os.path.join(base_dir, "model_gradboost.pkl")
with open(gb_model_file, 'wb') as f:
    pickle.dump(clf_gb, f)
print(f"[OK] Gradient Boosting model saved")

rf_model_file = os.path.join(base_dir, "model_randomforest.pkl")
with open(rf_model_file, 'wb') as f:
    pickle.dump(clf_rf, f)
print(f"[OK] Random Forest model saved")

scaler_file = os.path.join(base_dir, "scaler.pkl")
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"[OK] Scaler saved")

info_file = os.path.join(base_dir, "model_info.pkl")
with open(info_file, 'wb') as f:
    pickle.dump({
        'feature_names': X.columns.tolist(),
        'label_encoder': label_encoder,
        'xgb_accuracy': accuracy_xgb,
        'gradboost_accuracy': accuracy_gb,
        'randomforest_accuracy': accuracy_rf,
        'ensemble_accuracy': accuracy_ensemble
    }, f)
print(f"[OK] Model info saved")

# ============================================================================
# 📈 **VISUALIZATIONS**
# ============================================================================
print("\n📊 Generating visualizations...")

# 1. Confusion Matrix (Ensemble)
plt.figure(figsize=(8, 6))
cm_ensemble = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm_ensemble, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'Count'})
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - Ensemble Model (Acc: {accuracy_ensemble*100:.2f}%)")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
print("[OK] Confusion matrix saved")
plt.close()

# 2. Feature Importance (XGBoost)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf_xgb.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Top 15 Important EEG Channels (XGBoost)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
print("[OK] Feature importance saved")
plt.close()

# 3. Model Comparison Bar Chart
models = ['Baseline\n(74%)', 'XGBoost', 'GradBoost', 'RandomForest', 'Ensemble\n(Best)']
accuracies = [baseline, accuracy_xgb, accuracy_gb, accuracy_rf, accuracy_ensemble]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#96CEB4']

plt.figure(figsize=(11, 6))
bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)
plt.ylim([0.70, 0.90])
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('ADHD Detection Model Comparison - Ensemble Approach', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
print("[OK] Model comparison chart saved")
plt.close()

# 4. ROC-like comparison (prediction confidence)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_pred_ensemble_prob[y_test == 0], bins=50, alpha=0.6, label='ADHD (Class 0)', color='red')
plt.hist(y_pred_ensemble_prob[y_test == 1], bins=50, alpha=0.6, label='Control (Class 1)', color='blue')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('Ensemble Model - Prediction Confidence Distribution')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
# Precision-Recall for increasing thresholds
thresholds = np.arange(0.3, 0.8, 0.05)
accuracies_thresh = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_ensemble_prob > thresh).astype(int)
    acc_thresh = accuracy_score(y_test, y_pred_thresh)
    accuracies_thresh.append(acc_thresh)

plt.plot(thresholds, accuracies_thresh, marker='o', linewidth=2, markersize=8)
plt.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
plt.xlabel('Classification Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Decision Threshold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "prediction_analysis.png"), dpi=300, bbox_inches='tight')
print("[OK] Prediction analysis saved")
plt.close()

# ============================================================================
# SAVE DETAILED METRICS
# ============================================================================
metrics_file = os.path.join(base_dir, "model_metrics.txt")
with open(metrics_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ADHD DISEASE DETECTION - ENSEMBLE MODEL PERFORMANCE\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-"*70 + "\n")
    f.write(f"Total Samples: {len(df):,}\n")
    f.write(f"Training Samples: {len(X_train):,} (80%)\n")
    f.write(f"Testing Samples: {len(X_test):,} (20%)\n")
    f.write(f"Number of Features: {X.shape[1]} EEG channels\n")
    f.write(f"Classes: {len(np.unique(y))} (ADHD vs Control)\n\n")
    
    f.write("MODEL PERFORMANCE COMPARISON\n")
    f.write("-"*70 + "\n")
    f.write(f"Baseline Model (Original XGBoost): {baseline*100:.2f}%\n")
    f.write(f"Optimized XGBoost:                 {accuracy_xgb*100:.2f}% ({(accuracy_xgb-baseline)*100:+.2f}%)\n")
    f.write(f"Gradient Boosting:                 {accuracy_gb*100:.2f}% ({(accuracy_gb-baseline)*100:+.2f}%)\n")
    f.write(f"Random Forest:                     {accuracy_rf*100:.2f}% ({(accuracy_rf-baseline)*100:+.2f}%)\n")
    f.write(f"Ensemble Model (BEST):             {accuracy_ensemble*100:.2f}% ({(accuracy_ensemble-baseline)*100:+.2f}%)\n\n")
    
    f.write("DETAILED CLASSIFICATION REPORT (ENSEMBLE)\n")
    f.write("-"*70 + "\n")
    f.write(classification_report(y_test, y_pred_final, target_names=label_encoder.classes_))
    
    f.write("\nCONFUSION MATRIX (ENSEMBLE)\n")
    f.write("-"*70 + "\n")
    f.write(str(cm_ensemble) + "\n")
    f.write(f"Correct Predictions: {(y_pred_final == y_test).sum():,}\n")
    f.write(f"Incorrect Predictions: {(y_pred_final != y_test).sum():,}\n\n")
    
    f.write("="*70 + "\n")
    f.write("TRAINING COMPLETE\n")
    f.write("="*70 + "\n")

print("[OK] Detailed metrics saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("TRAINING COMPLETE - SUMMARY")
print("="*70)
print(f"\nBEST MODEL: Ensemble")
print(f"   Accuracy: {accuracy_ensemble*100:.2f}%")
print(f"   Change from baseline: {(accuracy_ensemble-baseline)*100:+.2f}%")
print(f"\nSaved Files:")
print(f"   - model_xgboost.pkl")
print(f"   - model_gradboost.pkl")
print(f"   - model_randomforest.pkl")
print(f"   - scaler.pkl")
print(f"   - model_info.pkl")
print(f"   - model_metrics.txt")
print(f"   - confusion_matrix.png")
print(f"   - feature_importance.png")
print(f"   - model_comparison.png")
print(f"   - prediction_analysis.png")
print("="*70 + "\n")
