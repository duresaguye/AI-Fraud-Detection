import flask
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging

# Initialize Flask application
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model
try:
    model = joblib.load('../models/fraud_random_forest.pkl')
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
        input_data = pd.DataFrame([data])
        logging.info(f"Input data: {input_data}")

        # Make prediction
        prediction = model.predict(input_data)
        logging.info(f"Prediction: {prediction}")

        # Return prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(debug=True)