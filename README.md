# AI-Fraud-Detection-e-commerce-and-bank-transactions

## Overview
This project aims to detect fraudulent activities in e-commerce and bank transactions using machine learning techniques. The datasets include e-commerce transaction data and bank transaction data specifically curated for fraud detection analysis.

## Datasets
### Fraud_Data.csv
Includes e-commerce transaction data aimed at identifying fraudulent activities.

- `user_id`: A unique identifier for the user who made the transaction.
- `signup_time`: The timestamp when the user signed up.
- `purchase_time`: The timestamp when the purchase was made.
- `purchase_value`: The value of the purchase in dollars.
- `device_id`: A unique identifier for the device used to make the transaction.
- `source`: The source through which the user came to the site (e.g., SEO, Ads).
- `browser`: The browser used to make the transaction (e.g., Chrome, Safari).
- `sex`: The gender of the user (M for male, F for female).
- `age`: The age of the user.
- `ip_address`: The IP address from which the transaction was made.
- `class`: The target variable where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.

### IpAddress_to_Country.csv
Maps IP addresses to countries.

- `lower_bound_ip_address`: The lower bound of the IP address range.
- `upper_bound_ip_address`: The upper bound of the IP address range.
- `country`: The country corresponding to the IP address range.

### creditcard.csv
Contains bank transaction data specifically curated for fraud detection analysis.

- `Time`: The number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` to `V28`: These are anonymized features resulting from a PCA transformation. Their exact nature is not disclosed for privacy reasons, but they represent the underlying patterns in the data.
- `Amount`: The transaction amount in dollars.
- `Class`: The target variable where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.

## Data Analysis and Preprocessing

### 1. Handle Missing Values
- Impute or drop missing values.

### 2. Data Cleaning
- Remove duplicates.
- Correct data types.

### 3. Exploratory Data Analysis (EDA)
- Univariate analysis.
- Bivariate analysis.

### 4. Merge Datasets for Geolocation Analysis
- Convert IP addresses to integer format.
- Merge `Fraud_Data.csv` with `IpAddress_to_Country.csv`.

### 5. Feature Engineering
- Transaction frequency and velocity for `Fraud_Data.csv`.
- Time-Based features for `Fraud_Data.csv`:
    - `hour_of_day`
    - `day_of_week`

### 6. Normalization and Scaling
- Normalize and scale numerical features.

### 7. Encode Categorical Features
- Encode categorical features using label encoding.

## Visualizations
### Distribution of Purchase Value
![Distribution of Purchase Value](visualization/purchaseValue.png)


### Correlation Matrix
![Correlation Matrix](visualization/correlation.png)

#### Building and Deploying a Fraud Detection Model: A Journey from Data to API

In today's digital landscape, fraud detection is a critical task for businesses of all sizes. The ability to identify and prevent fraudulent transactions in real-time can save significant amounts of money and protect brand reputation. In this article, I'll walk you through the process of building and deploying a fraud detection model, highlighting the challenges and solutions encountered along the way.

**The Data:**

Our journey begins with a dataset containing transaction information, including user IDs, timestamps, purchase values, device information, and various behavioral features. The goal is to train a machine learning model that can accurately predict whether a given transaction is fraudulent or legitimate.

**Feature Engineering:**

The first step is feature engineering, where we transform the raw data into a format suitable for machine learning. This involves:

*   **Time-Based Features:** Extracting temporal information from the `signup_time` and `purchase_time` columns, such as year, month, day, and hour. These features can capture patterns related to the time of day or year when fraudulent activities are more likely to occur.
*   **Categorical Encoding:** Converting categorical features like `source`, `browser`, and `sex` into numerical representations using label encoding.
*   **Feature Scaling:** Scaling numerical features using `StandardScaler` to ensure that all features contribute equally to the model's learning process.

**Model Training:**

With the features engineered, we can now train a machine learning model. A `RandomForestClassifier` was chosen for its ability to handle complex relationships and provide relatively good performance without extensive hyperparameter tuning. The model is trained on the preprocessed data, and its performance is evaluated using appropriate metrics like precision, recall, and F1-score.

**Deployment with Flask:**

To make the model accessible, we deploy it as a REST API using Flask, a lightweight Python web framework. The Flask application exposes a `/predict` endpoint that accepts transaction data as JSON input, preprocesses the data in the same way as during training, and returns the model's prediction.

**Challenges and Solutions:**

The deployment process wasn't without its challenges. One of the main hurdles was ensuring consistency between the training and serving environments. This involved:

*   **Feature Order:** Maintaining the correct order of features during prediction to match the order used during training.
*   **Data Preprocessing:** Applying the same preprocessing steps (time feature extraction, categorical encoding, scaling) in the serving script as were used during training.
*   **Dependency Management:** Ensuring that all necessary libraries and dependencies are available in the serving environment.

These challenges were addressed through careful attention to detail and thorough testing.

**The `serve_model.py` Script:**

The core of the deployment is the `serve_model.py` script, which handles the API requests and model predictions. Here's a simplified version of the script:

```python
import flask
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'model/fraud_random_forest_model.pkl'
SCALER_PATH = 'model/fraud_scaler.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading the model or scaler: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # Preprocessing steps (time features, encoding, scaling)

        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

## How to Run
Setup Instructions

    Clone the Repository

git clone https://github.com/duresaguye/AI-Fraud-Detection


cd AI-Fraud-Detection

Install Dependencies
Ensure all required dependencies are installed:

pip install -r requirements.txt

Run Data Preprocessing
Execute the Jupyter notebook to preprocess the data and generate visualizations:

    Open the notebooks folder.
    Run Data_Preprocessing_with_Scikit_Learn.ipynb in Jupyter Notebook.
