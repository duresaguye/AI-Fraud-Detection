import flask
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model path
MODEL_PATH = "model/creditcard_fraud_rf_model.pkl"  

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        logging.info(f"Received data: {data}")

        # Convert data to DataFrame
        input_data = pd.DataFrame([data], index=[0])  # Create DataFrame with index

        # Extract time-based features
        input_data['purchase_time'] = pd.to_datetime(input_data['purchase_time'])
        input_data['signup_time'] = pd.to_datetime(input_data['signup_time'])

        input_data['purchase_time_year'] = input_data['purchase_time'].dt.year
        input_data['purchase_time_month'] = input_data['purchase_time'].dt.month
        input_data['purchase_time_day'] = input_data['purchase_time'].dt.day
        input_data['purchase_time_hour'] = input_data['purchase_time'].dt.hour

        input_data['signup_time_year'] = input_data['signup_time'].dt.year
        input_data['signup_time_month'] = input_data['signup_time'].dt.month
        input_data['signup_time_day'] = input_data['signup_time'].dt.day
        input_data['signup_time_hour'] = input_data['signup_time'].dt.hour

        # Remove original time columns
        input_data = input_data.drop(columns=['purchase_time', 'signup_time', 'device_id'])

        # Encode categorical features
        label_encoder = LabelEncoder()
        input_data['source'] = label_encoder.fit_transform(input_data['source'])
        input_data['browser'] = label_encoder.fit_transform(input_data['browser'])
        input_data['sex'] = label_encoder.fit_transform(input_data['sex'])

        
        correct_column_order = [
            'user_id',
            'purchase_value',
            'source',
            'browser',
            'sex',
            'age',
            'ip_address',
            'transaction_frequency',
            'transaction_velocity',
            'hour_of_day',
            'day_of_week',
            'purchase_time_year',
            'purchase_time_month',
            'purchase_time_day',
            'purchase_time_hour',
            'signup_time_year',
            'signup_time_month',
            'signup_time_day',
            'signup_time_hour'
        ]
        input_data = input_data[correct_column_order]

        # Make prediction
        prediction = model.predict(input_data)
        logging.info(f"Prediction: {prediction}")

        # Return prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)