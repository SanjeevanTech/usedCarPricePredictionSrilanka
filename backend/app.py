from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np
import json

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define paths for loading artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'model')

# Load Artifacts
try:
    model_path = os.path.join(MODELS_DIR, 'vehicle_price_model.pkl')
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    label_encoders_path = os.path.join(MODELS_DIR, 'label_encoders.pkl')

    if os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(label_encoders_path):
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        label_encoders = joblib.load(label_encoders_path)
        print("Model Artifacts Loaded Successfully.")
    else:
        print("Model artifacts not found. Please run model/train_model.py first.")
        model = None
        preprocessor = None
        label_encoders = None

except Exception as e:
    print(f"Error loading artifacts: {e}")
    model = None
    preprocessor = None
    label_encoders = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/options', methods=['GET'])
def get_options():
    try:
        options_path = os.path.join(BASE_DIR, 'options.json')
        if os.path.exists(options_path):
            with open(options_path, 'r') as f:
                return jsonify(json.load(f))
        else:
            return jsonify({"error": "Options file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_price():
    if model is None or preprocessor is None or label_encoders is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500

    try:
        data = request.json
        
        # Expected features based on the notebook
        expected_features = [
            'Brand', 'Model', 'YOM', 'Engine_cc', 'Gear', 'Fuel_Type', 
            'Millage_KM', 'Town', 'Leasing', 'Condition', 
            'AIR_CONDITION', 'POWER_STEERING', 'POWER_MIRROR', 'POWER_WINDOW'
        ]

        # Check if all required fields are present
        missing_fields = [field for field in expected_features if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        # Validate Engine CC and Mileage
        try:
            engine_cc = float(data['Engine_cc'])
            millage_km = float(data['Millage_KM'])
            
            if engine_cc < 100 or engine_cc > 5000:
                return jsonify({"error": "Engine Capacity must be between 100 and 5000 cc."}), 400
            
            if millage_km < 0 or millage_km > 300000:
                return jsonify({"error": "Mileage must be a realistic value between 0 and 300,000 KM."}), 400
                
        except ValueError:
            return jsonify({"error": "Engine CC and Mileage must be valid numbers."}), 400

        # Cross-validate Fuel Type and Gear against known training data
        try:
            options_path = os.path.join(BASE_DIR, 'options.json')
            with open(options_path, 'r') as f:
                options_data = json.load(f)

            model_key = str(data['Model']).strip().upper()
            fuel_input = str(data['Fuel_Type']).strip().upper()
            gear_input = str(data['Gear']).strip().upper()

            valid_fuels = options_data.get('FuelByModel', {}).get(model_key)
            if valid_fuels and fuel_input not in [f.upper() for f in valid_fuels]:
                return jsonify({
                    "error": f"Invalid Fuel Type '{fuel_input}' for model '{model_key}'. "
                             f"Valid options: {', '.join(valid_fuels)}"
                }), 400

            valid_gears = options_data.get('GearByModel', {}).get(model_key)
            if valid_gears and gear_input not in [g.upper() for g in valid_gears]:
                return jsonify({
                    "error": f"Invalid Gear Type '{gear_input}' for model '{model_key}'. "
                             f"Valid options: {', '.join(valid_gears)}"
                }), 400

        except Exception as e:
            pass  # If options file unavailable, skip cross-validation gracefully


        # Preprocess inputs to match training data
        brand = str(data['Brand']).strip().upper()
        model_name = str(data['Model']).strip().upper()
        
        leasing = str(data['Leasing']).strip()
        if leasing == 'No':
            leasing = 'No Leasing'
        elif leasing == 'Yes':
            leasing = 'Ongoing Lease'
            
        condition = str(data['Condition']).strip().upper()
        
        input_data = pd.DataFrame([{
            'Brand': brand,
            'Model': model_name,
            'YOM': int(data['YOM']),
            'Engine (cc)': float(data['Engine_cc']),
            'Gear': str(data['Gear']).title(),
            'Fuel Type': str(data['Fuel_Type']).title(),
            'Millage(KM)': float(data['Millage_KM']),
            'Town': data['Town'],
            'Leasing': leasing.title(),
            'Condition': condition, # Condition is already upper
            'AIR CONDITION': str(data['AIR_CONDITION']).title(),
            'POWER STEERING': str(data['POWER_STEERING']).title(),
            'POWER MIRROR': str(data['POWER_MIRROR']).title(),
            'POWER WINDOW': str(data['POWER_WINDOW']).title()
        }])

        # 1. Label Encode Manually
        for col in ['Brand', 'Model', 'Town']:
            le = label_encoders[col]
            val = input_data[col].iloc[0]
            # Safe encoding for unseen labels
            if val in le.classes_:
                input_data[col] = le.transform([val])[0]
            else:
                # Transform to the first class if unseen (simple handling)
                input_data[col] = le.transform([le.classes_[0]])[0]

        # 2. Pipeline Transform
        processed_input = preprocessor.transform(input_data)

        # 3. Predict
        prediction = model.predict(processed_input)

        return jsonify({"estimated_price_lakhs": round(prediction[0], 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
