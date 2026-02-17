# ADHD Disease Detection - Ensemble Learning Model

## Overview

This project implements an advanced machine learning system for detecting Attention-Deficit/Hyperactivity Disorder (ADHD) using ensemble learning techniques. The system combines three powerful algorithms (XGBoost, Gradient Boosting, and Random Forest) through soft voting to achieve superior classification accuracy.

## Project Features

✨ **Advanced Ensemble Approach**
- XGBoost with optimized hyperparameters
- Gradient Boosting classifier
- Random Forest with balanced class weights
- Weighted soft voting ensemble (40% XGBoost, 30% GradBoost, 30% RandomForest)

🎯 **High Performance**
- Exceeds baseline accuracy by significant margin
- Handles imbalanced dataset with class weighting
- Cross-validated and stratified train-test split
- Optimized for both speed and accuracy

📊 **Comprehensive Evaluation**
- Detailed classification reports
- Confusion matrix analysis
- Feature importance visualization
- Prediction confidence distribution analysis
- Model comparison charts

## Requirements

```
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=1.7.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd ADHD_Disease_Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the dataset is in the correct location:
```
ADHD_Disease_Detection/
├── Dataset/
│   └── adhdata.csv
├── main.py
└── ...
```

## Usage

### Training the Model

Run the main training script:
```bash
python main.py
```

This will:
- Load and preprocess the ADHD dataset
- Train all three models (XGBoost, Gradient Boosting, Random Forest)
- Create an ensemble voting classifier
- Generate evaluation metrics and visualizations
- Save trained models and artifacts

### Viewing Results

After training, check the generated files:

**Visualizations:**
- `confusion_matrix.png` - Ensemble model confusion matrix
- `feature_importance.png` - Top 15 important EEG channels
- `model_comparison.png` - Accuracy comparison across all models
- `prediction_analysis.png` - Prediction confidence and threshold analysis

**Model Files:**
- `model_xgboost.pkl` - Trained XGBoost model
- `model_gradboost.pkl` - Trained Gradient Boosting model
- `model_randomforest.pkl` - Trained Random Forest model
- `scaler.pkl` - StandardScaler for feature normalization
- `model_info.pkl` - Feature names and label encoder
- `model_metrics.txt` - Detailed performance metrics

### Testing and Viewing Accuracy

```bash
# View model accuracy
python view_accuracy.py

# Test model accuracy
python test_accuracy.py
```

## Model Architecture

### 1. **XGBoost (Optimized)**
- 150 estimators with max depth of 5
- Learning rate: 0.1
- Regularization: L1 (0.5) + L2 (1.0)
- Subsample & column sampling for robustness
- GPU-accelerated training with 'hist' tree method

### 2. **Gradient Boosting**
- 50 estimators with max depth of 4
- Learning rate: 0.1
- 70% subsampling for faster training

### 3. **Random Forest**
- 100 estimators with max depth of 10
- Balanced class weights to handle ADHD vs Control imbalance
- sqrt feature selection
- Min samples split: 15, Min samples leaf: 8

### 4. **Ensemble (Best Model)**
- Weighted soft voting using predicted probabilities
- Final prediction based on 0.5 threshold
- Weights optimized through grid search

## Dataset

- **Source**: EEG-based ADHD detection dataset
- **Features**: Multiple EEG channels (numeric features)
- **Target**: Binary classification (ADHD vs Control)
- **Preprocessing**: 
  - Missing value removal
  - Feature standardization
  - Stratified train-test split (80-20)

## Model Performance

The ensemble model achieves superior performance through:

1. **Individual Model Strengths**: Each model captures different patterns in EEG data
2. **Weighted Voting**: Leverages the strengths of each algorithm based on contribution
3. **Probability Averaging**: Uses soft voting for more nuanced decision making
4. **Threshold Optimization**: Adjustable classification threshold for different use cases

## Files Structure

```
ADHD_Disease_Detection/
├── main.py                 # Main training script
├── test_accuracy.py        # Test the trained model
├── view_accuracy.py        # View accuracy results
├── requirements.txt        # Python dependencies
├── model_metrics.txt       # Performance metrics
├── adhdata.csv            # Input dataset
├── README.md              # This file
│
├── Models & Artifacts:
├── model_xgboost.pkl      # XGBoost model
├── model_gradboost.pkl    # Gradient Boosting model
├── model_randomforest.pkl # Random Forest model
├── scaler.pkl             # Feature scaler
├── model_info.pkl         # Model metadata
│
└── Visualizations:
    ├── confusion_matrix.png      # Confusion matrix heatmap
    ├── feature_importance.png    # Top EEG channel importance
    ├── model_comparison.png      # Model accuracy comparison
    └── prediction_analysis.png   # Confidence & threshold analysis
```

## Key Features

📈 **Data Processing**
- Automatic handling of missing values
- StandardScaler normalization
- Feature selection (numeric features only)
- Stratified sampling for balanced splits

🤖 **Model Training**
- Hyperparameter optimization
- Cross-validation support
- Class weight balancing
- Early stopping capabilities

📊 **Visualization**
- Heatmap confusion matrices
- Feature importance bar charts
- Model comparison bar plots
- Prediction confidence histograms
- Threshold sensitivity analysis

## Performance Metrics

The models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Type I error control
- **Recall**: Type II error control
- **F1-Score**: Harmonic mean for imbalanced data
- **Confusion Matrix**: Detailed prediction breakdown

## Future Improvements

- Deep learning approaches (LSTM, CNN for temporal EEG data)
- Hyperparameter tuning with Bayesian optimization
- Cross-validation ensemble voting
- Real-time prediction API
- Mobile deployment

## Notes

- Models are saved in pickle format for easy loading and inference
- All visualizations are high-resolution (300 DPI) for presentations
- The dataset is processed in-memory with up to 500K samples for training speed
- Compatible with Python 3.8+

## License

This project is for research and educational purposes.

## Contact

For questions or issues, please reach out to the project maintainers.