import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataCleaner:
    @staticmethod
    def handle_missing_values(df, fill_method='median'):
        if fill_method == 'median':
            return df.fillna(df.median())
        elif fill_method == 'mean':
            return df.fillna(df.mean())
        else:
            return df.dropna()
    
    @staticmethod
    def remove_duplicates(df):
        return df.drop_duplicates()
    
    @staticmethod
    def convert_data_types(df, datetime_cols):
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col])
        return df

class EDA:
    @staticmethod
    def univariate_analysis(df):
        print(df.describe())
    
    @staticmethod
    def bivariate_analysis(df, group_by_col, target_col):
        print(df.groupby(group_by_col)[target_col].mean())

class FeatureEngineer:
    @staticmethod
    def transaction_features(df):
        df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
        return df
    
    @staticmethod
    def time_based_features(df, time_col):
        df['hour_of_day'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.dayofweek
        return df

class DataPreprocessor:
    def __init__(self, fraud_data_path, ip_data_path, credit_card_data_path):
        self.fraud_data = pd.read_csv(fraud_data_path)
        self.ip_data = pd.read_csv(ip_data_path)
        self.credit_card_data = pd.read_csv(credit_card_data_path)

    def preprocess_fraud_data(self):
        self.fraud_data = DataCleaner.handle_missing_values(self.fraud_data)
        self.fraud_data = DataCleaner.remove_duplicates(self.fraud_data)
        self.fraud_data = DataCleaner.convert_data_types(self.fraud_data, ['signup_time', 'purchase_time'])
        
        EDA.univariate_analysis(self.fraud_data)
        EDA.bivariate_analysis(self.fraud_data, 'class', 'age')
        
        self.fraud_data = FeatureEngineer.transaction_features(self.fraud_data)
        self.fraud_data = FeatureEngineer.time_based_features(self.fraud_data, 'purchase_time')
        
        return self.fraud_data

    def preprocess_credit_card_data(self):
        self.credit_card_data = DataCleaner.handle_missing_values(self.credit_card_data)
        self.credit_card_data = DataCleaner.remove_duplicates(self.credit_card_data)
        
        EDA.univariate_analysis(self.credit_card_data)
        return self.credit_card_data

    def preprocess(self):
        fraud_data = self.preprocess_fraud_data()
        credit_card_data = self.preprocess_credit_card_data()
        return fraud_data, credit_card_data

# Usage
if __name__ == "__main__":
    processor = DataPreprocessor("Fraud_Data.csv", "IpAddress_to_Country.csv", "creditcard.csv")
    fraud_data, credit_card_data = processor.preprocess()
    
    fraud_data.to_csv("Preprocessed_Fraud_Data.csv", index=False)
    credit_card_data.to_csv("Preprocessed_CreditCard_Data.csv", index=False)
