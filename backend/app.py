from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Directory for saving models
MODELS_DIR = 'saved_models'
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Global variables to store current model and scaler
current_model = None
current_scaler = None
current_model_name = None
model_metrics = None
uploaded_file_data = None  # Store uploaded file data for training

# Available models
MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Upload and preview training data"""
    global uploaded_file_data
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Validate data
        if df.empty:
            return jsonify({'error': 'File is empty'}), 400
        
        # Store FULL data globally for training
        uploaded_file_data = df
        
        return jsonify({
            'message': 'File uploaded successfully',
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'preview': df.head(5).to_dict(orient='records'),
            'data_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict()
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train multiple models"""
    global current_model, current_scaler, current_model_name, model_metrics, uploaded_file_data
    
    try:
        if uploaded_file_data is None:
            return jsonify({'error': 'No data uploaded. Please upload data first.'}), 400
        
        data = request.json
        
        # Use the stored full data, not the preview
        df = uploaded_file_data.copy()
        
        # Get target column and features
        target_col = data.get('target_column', df.columns[-1])
        model_names = data.get('models', list(MODELS.keys()))
        
        if target_col not in df.columns:
            return jsonify({'error': f'Target column "{target_col}" not found'}), 400
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Ensure target is numeric (0 or 1)
        if y.dtype == 'object':
            y = pd.factorize(y)[0]
        
        # Check if we have both classes
        unique_classes = len(y.unique())
        if unique_classes < 2:
            return jsonify({'error': f'Data must have at least 2 fault classes, but only found {unique_classes}. Check that your data has both fault=0 and fault=1 samples.'}), 400
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train all selected models and get metrics
        results = {}
        best_model_name = None
        best_accuracy = 0
        best_model = None
        
        for model_name in model_names:
            if model_name not in MODELS:
                continue
            
            model = MODELS[model_name]
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            results[model_name] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm
            }
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                best_model = model
        
        # Save best model
        if best_model:
            current_model = best_model
            current_scaler = scaler
            current_model_name = best_model_name
            model_metrics = results
            
            # Save to disk
            model_path = os.path.join(MODELS_DIR, f'{best_model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
            joblib.dump(best_model, model_path)
            joblib.dump(scaler, model_path.replace('.pkl', '_scaler.pkl'))
        
        return jsonify({
            'message': 'Training completed',
            'best_model': best_model_name,
            'metrics': results,
            'data_split': {
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on new data"""
    global current_model, current_scaler, current_model_name
    
    try:
        if current_model is None or current_scaler is None:
            return jsonify({'error': 'No model trained yet. Please train a model first.'}), 400
        
        data = request.json
        
        if not data or 'input_data' not in data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data['input_data']])
        
        # Scale input
        df_scaled = current_scaler.transform(df)
        
        # Make prediction
        prediction = current_model.predict(df_scaled)[0]
        probability = current_model.predict_proba(df_scaled)[0] if hasattr(current_model, 'predict_proba') else None
        
        result = {
            'prediction': int(prediction),
            'status': 'Fault Detected' if prediction == 1 else 'No Fault',
            'model_used': current_model_name,
            'probability': [float(p) for p in probability] if probability is not None else None
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction on multiple samples"""
    global current_model, current_scaler, current_model_name
    
    try:
        if current_model is None or current_scaler is None:
            return jsonify({'error': 'No model trained yet. Please train a model first.'}), 400
        
        data = request.json
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Scale input
        df_scaled = current_scaler.transform(df)
        
        # Make predictions
        predictions = current_model.predict(df_scaled)
        probabilities = current_model.predict_proba(df_scaled) if hasattr(current_model, 'predict_proba') else None
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'sample': i,
                'prediction': int(pred),
                'status': 'Fault Detected' if pred == 1 else 'No Fault',
                'probability': [float(p) for p in probabilities[i]] if probabilities is not None else None
            }
            results.append(result)
        
        return jsonify({
            'predictions': results,
            'model_used': current_model_name,
            'total_faults': int(sum(predictions)),
            'fault_percentage': float((sum(predictions) / len(predictions)) * 100)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get current model information"""
    global current_model, current_model_name, model_metrics
    
    if current_model is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    return jsonify({
        'model_name': current_model_name,
        'metrics': model_metrics,
        'available_models': list(MODELS.keys())
    }), 200

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'Server is running'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
