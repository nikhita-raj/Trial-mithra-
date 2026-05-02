from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from chatbot import TrialMitraChatbot
import numpy as np

app = Flask(__name__)

# Initialize chatbot
chatbot = TrialMitraChatbot()

def load_global_model():
    """Load the latest global federated model"""
    try:
        with open('models/global_model_latest.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        return None

def predict_eligibility(patient_data, model_data):
    """Predict patient eligibility using the global model"""
    if model_data is None:
        return None
    
    # Prepare patient data
    patient_df = pd.DataFrame([patient_data])
    
    # Encode categorical variables (simplified)
    gender_map = {'Male': 1, 'Female': 0}
    disease_map = {
        'diabetes': 0, 'hypertension': 1, 'heart_disease': 2, 
        'cancer': 3, 'mental_health': 4
    }
    
    patient_df['gender'] = patient_df['gender'].map(gender_map)
    patient_df['disease'] = patient_df['disease'].map(disease_map)
    
    # Scale features (simplified scaling)
    feature_cols = ['age', 'gender', 'bp', 'sugar', 'disease', 'trial_interest', 'engagement_level']
    X = patient_df[feature_cols].values
    
    # Simple normalization
    X_normalized = (X - X.mean()) / (X.std() + 1e-8)
    
    # Predict using model parameters
    coef = model_data['coef_']
    intercept = model_data['intercept_']
    
    # Logistic regression prediction
    z = np.dot(X_normalized, coef.T) + intercept
    probability = 1 / (1 + np.exp(-z))
    prediction = (probability > 0.5).astype(int)
    
    return {
        'eligible': bool(prediction[0][0]),
        'probability': float(probability[0][0]),
        'confidence': float(max(probability[0][0], 1 - probability[0][0]))
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Prepare patient data
        patient_data = {
            'age': int(data['age']),
            'gender': data['gender'],
            'bp': int(data['bp']),
            'sugar': int(data['sugar']),
            'disease': data['disease'],
            'trial_interest': int(data['trial_interest']),
            'engagement_level': int(data['engagement_level'])
        }
        
        # Load model and predict
        model_data = load_global_model()
        
        if model_data:
            result = predict_eligibility(patient_data, model_data)
            
            # Get matching trials
            matching_trials = chatbot.check_basic_eligibility(
                patient_data['age'], 
                patient_data['bp'], 
                patient_data['disease']
            )
            
            return jsonify({
                'success': True,
                'prediction': result,
                'matching_trials': [trial.to_dict() for trial in matching_trials]
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Federated model not available'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data['message']
        patient_id = data.get('patient_id', 'WEB001')
        age = data.get('age', 45)
        bp = data.get('bp', 120)
        disease = data.get('disease', 'diabetes')
        
        response = chatbot.generate_response(
            user_input, 
            patient_id=patient_id, 
            age=age, 
            bp=bp, 
            disease=disease
        )
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/trials')
def trials():
    try:
        trials_df = pd.read_csv('data/clinical_trials.csv')
        return jsonify({
            'success': True,
            'trials': trials_df.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/dashboard')
def dashboard():
    try:
        # Load federated learning results
        results_df = pd.read_csv('results/federated_results.csv')
        
        # Load hospital data
        h1_df = pd.read_csv('data/hospital1_patients_cleaned.csv')
        h2_df = pd.read_csv('data/hospital2_patients_cleaned.csv')
        h3_df = pd.read_csv('data/hospital3_patients_cleaned.csv')
        h4_df = pd.read_csv('data/hospital4_patients_cleaned.csv')
        
        dashboard_data = {
            'federated_results': results_df.to_dict('records'),
            'hospital_stats': {
                'hospital1': {
                    'total_patients': len(h1_df),
                    'eligible_patients': int(h1_df['eligible'].sum()),
                    'eligibility_rate': float(h1_df['eligible'].mean())
                },
                'hospital2': {
                    'total_patients': len(h2_df),
                    'eligible_patients': int(h2_df['eligible'].sum()),
                    'eligibility_rate': float(h2_df['eligible'].mean())
                },
                'hospital3': {
                    'total_patients': len(h3_df),
                    'eligible_patients': int(h3_df['eligible'].sum()),
                    'eligibility_rate': float(h3_df['eligible'].mean())
                },
                'hospital4': {
                    'total_patients': len(h4_df),
                    'eligible_patients': int(h4_df['eligible'].sum()),
                    'eligibility_rate': float(h4_df['eligible'].mean())
                }
            }
        }
        
        return jsonify({
            'success': True,
            'data': dashboard_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)