"""
Test script to demonstrate how accuracy changes with different dataset sizes
This helps answer: Why 40 samples give 100% but what happens with 100, 200, etc.?
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_circuit_data(n_samples):
    """Generate synthetic circuit fault data"""
    # Circuit parameters: voltage, current, temperature, resistance, frequency, power
    
    # Normal circuits (fault=0)
    n_normal = n_samples // 2
    normal_voltage = np.random.normal(230, 5, n_normal)      # ~230V
    normal_current = np.random.normal(10, 1, n_normal)       # ~10A
    normal_temp = np.random.normal(35, 3, n_normal)          # ~35°C
    normal_resistance = np.random.normal(23, 2, n_normal)    # ~23 Ohms
    normal_frequency = np.random.normal(50, 0.5, n_normal)   # ~50Hz
    normal_power = normal_voltage * normal_current           # P = V * I
    
    normal_data = np.column_stack([
        normal_voltage, normal_current, normal_temp, 
        normal_resistance, normal_frequency, normal_power
    ])
    normal_labels = np.zeros(n_normal)
    
    # Faulty circuits (fault=1) - anomalies
    n_faulty = n_samples - n_normal
    faulty_voltage = np.random.normal(240, 15, n_faulty)      # Higher variation, occasional spikes
    faulty_current = np.random.normal(15, 3, n_faulty)        # Higher current than normal
    faulty_temp = np.random.normal(55, 8, n_faulty)           # Much higher temperature
    faulty_resistance = np.random.normal(20, 3, n_faulty)     # Slightly lower resistance (shorts)
    faulty_frequency = np.random.normal(49, 1.5, n_faulty)    # More frequency variation
    faulty_power = faulty_voltage * faulty_current
    
    faulty_data = np.column_stack([
        faulty_voltage, faulty_current, faulty_temp,
        faulty_resistance, faulty_frequency, faulty_power
    ])
    faulty_labels = np.ones(n_faulty)
    
    # Combine
    X = np.vstack([normal_data, faulty_data])
    y = np.hstack([normal_labels, faulty_labels])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

def test_model_accuracy(X, y, model_name, model):
    """Test model and return metrics"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 80-20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'test_samples': len(y_test)
    }

# Test with different dataset sizes
dataset_sizes = [40, 100, 200, 500, 1000]
models_to_test = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

print("=" * 90)
print("CIRCUIT FAULT DETECTION: ACCURACY vs DATASET SIZE")
print("=" * 90)
print()

results = {}

for size in dataset_sizes:
    print(f"\n{'='*90}")
    print(f"DATASET SIZE: {size} samples ({size//2} normal + {size//2} faulty)")
    print(f"{'='*90}")
    
    # Generate data
    X, y = generate_circuit_data(size)
    print(f"Generated data shape: {X.shape}")
    print(f"Training set: {int(size * 0.8)} samples | Test set: {int(size * 0.2)} samples")
    print()
    
    size_results = {}
    
    for model_name, model in models_to_test.items():
        metrics = test_model_accuracy(X, y, model_name, model)
        size_results[model_name] = metrics
        
        print(f"  {model_name:25} | Accuracy: {metrics['accuracy']:.1%} | "
              f"Precision: {metrics['precision']:.1%} | Recall: {metrics['recall']:.1%} | "
              f"F1: {metrics['f1']:.1%}")
    
    results[size] = size_results

# Summary and Analysis
print(f"\n\n{'='*90}")
print("SUMMARY: ACCURACY TREND ANALYSIS")
print(f"{'='*90}\n")

for model_name in models_to_test.keys():
    print(f"\n{model_name}:")
    print(f"  Dataset Size | Accuracy")
    print(f"  " + "-" * 30)
    
    accuracies = []
    for size in dataset_sizes:
        acc = results[size][model_name]['accuracy']
        accuracies.append(acc)
        bar_length = int(acc * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {size:4} samples | {acc:6.1%} | {bar}")

print(f"\n\n{'='*90}")
print("KEY INSIGHTS:")
print(f"{'='*90}")
print("""
1. WHY 40 SAMPLES GIVES 100% ACCURACY:
   - Small dataset = Easy for models to memorize patterns
   - The differences between normal and faulty are very clear
   - 80-20 split = only 8 test samples per class
   - Random chance: With clear patterns, small samples often achieve high accuracy

2. WHAT HAPPENS WITH LARGER DATASETS:
   - More diverse data patterns emerge
   - Models can NOT memorize everything
   - Accuracy may decrease because real-world complexity appears
   - Test set is larger = more representative evaluation
   - More reliable performance metrics

3. IS 100% OVERFITTING?
   - NOT necessarily - depends on data quality
   - If data is very clean and patterns are distinct: High accuracy is valid
   - Decision Tree often overfit (can memorize) - check its accuracy vs others
   - Random Forest is more robust - compares well to others

4. BEST PRACTICES:
   - Larger datasets (100+) give more realistic accuracy estimates
   - If accuracy DROPS significantly: Models were overfitting on small data
   - If accuracy STAYS high: Good generalization, reliable model
   - Add noise/variation to test real-world robustness

5. FOR YOUR PRESENTATION:
   - Show this test to prove you understand ML concepts
   - Explain why accuracy changes with data size
   - Discuss overfitting vs good generalization
   - Demonstrate model robustness with larger datasets
""")
