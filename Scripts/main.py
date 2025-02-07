# main.py
from data_loader import load_data, initial_checks
from preprocessor import preprocess_fraud_data, preprocess_creditcard
from geo_merger import merge_with_ip_country
from feature_engineer import engineer_features
from eda import univariate_analysis, bivariate_analysis
import pandas as pd

def main():
    # Load and preprocess data
    fraud_data = load_data('fraud_data')
    fraud_data = preprocess_fraud_data(fraud_data)
    
    ip_country = load_data('ip_country')
    merged_data = merge_with_ip_country(fraud_data, ip_country)
    
    # Feature engineering
    merged_data = engineer_features(merged_data)
    
    # EDA
    univariate_analysis(merged_data, 'purchase_value', plot_type='box')
    bivariate_analysis(merged_data, 'source')

if __name__ == "__main__":
    main()