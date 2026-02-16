<<<<<<< HEAD
# ADHD Classification Pipeline
=======
# ADHD classification
# This is our 2nd project

>>>>>>> 4d85d944b638838964ae868266376e13c5816ec3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 1️⃣ **Load Dataset**
file_path = "adhdata.csv"  # Change to your actual file path
df = pd.read_csv(file_path)
# 2️⃣ **Check Dataset Info**
print(df.head())  # Check first few rows
print(df.info())  # Check column names & missing values
print(df.describe())  # Check statistics
# 3️⃣ **Handle Missing Values (If Any)**
df.dropna(inplace=True)
# 4️⃣ **Feature Selection (Exclude Non-Numeric Columns)**
X = df.select_dtypes(include=[np.number])  # Keep only numeric columns

# List the available columns for inspection
print(df.columns)

# Ask the user to input the correct target column name
target_column = input("Please enter the name of the target column: ")

# Try to access the target column using the user-provided name
try:
    y = df[target_column]  # Target column
except KeyError:
    raise KeyError(f"Target column '{target_column}' not found in the DataFrame. Please check your data and try again.")
# 5️⃣ **Normalize Data**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Ensure matching shapes
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (with stratification if possible)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as e:
    print(f"⚠ Warning: {e} (Stratify removed)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

# Check final shapes
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
# 7️⃣ **Train a RandomForest Model (Best for Tabular Data)**
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# 8️⃣ **Make Predictions**
y_pred = clf.predict(X_test)
# 9️⃣ **Evaluate Model**
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Model Accuracy: {accuracy:.4f}")
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# 1️⃣0️⃣ **Confusion Matrix**
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
# 1️⃣1️⃣ **Feature Importance Visualization**
feature_importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
feature_importance.plot(kind="bar", color="teal")
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.show()
# 1️⃣2️⃣ **Plot Waveforms of Actual vs. Predicted**
def plot_waveform(y_true, y_pred, sample_indices):
    plt.figure(figsize=(12, 6))

    plt.plot(sample_indices, y_true, label="Actual", linestyle="dashed", marker="o", color="blue")
    plt.plot(sample_indices, y_pred, label="Predicted", linestyle="solid", marker="x", color="red")

    plt.xlabel("Sample Index")
    plt.ylabel("Condition (0 = Normal, 1 = ADHD)")
    plt.title("Waveform of Actual vs. Predicted ADHD Classification")
    plt.legend()
    plt.grid(True)
    plt.show()
# Get 20 random samples for visualization
sample_indices = np.arange(len(y_test[:20]))
plot_waveform(y_test[:20].values, y_pred[:20], sample_indices)
import numpy as np
import matplotlib.pyplot as plt
# Function to plot Actual vs. Predicted Waveform
def plot_waveform(y_true, y_pred, sample_indices):
    plt.figure(figsize=(12, 6))

    plt.plot(sample_indices, y_true, label="Actual", linestyle="dashed", marker="o", color="blue", markersize=6)
    plt.plot(sample_indices, y_pred, label="Predicted", linestyle="solid", marker="x", color="red", markersize=6)

    plt.xlabel("Sample Index")
    plt.ylabel("Condition (0 = Normal, 1 = ADHD)")
    plt.title("Waveform of Actual vs. Predicted ADHD Classification")
    plt.legend()
    plt.grid(True)
    plt.show()
# 1️⃣ **Get 20 Random Samples for Visualization**
random_indices = np.random.choice(len(y_test), 20, replace=False)  # Pick 20 random samples
y_true_samples = y_test.iloc[random_indices].values  # Get corresponding actual values
y_pred_samples = y_pred[random_indices]  # Get predicted values
plot_waveform(y_true_samples, y_pred_samples, np.arange(20))  # Plot waveforms



