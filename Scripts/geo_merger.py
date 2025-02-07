
import pandas as pd
from utils import ip_to_int

def merge_with_ip_country(fraud_df, ip_country_df):
    # Convert IPs to integers
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)
    ip_country_df['lower_int'] = ip_country_df['lower_bound_ip_address'].apply(ip_to_int)
    ip_country_df['upper_int'] = ip_country_df['upper_bound_ip_address'].apply(ip_to_int)
    
    # Merge using interval conditions
    merged_df = pd.merge(
        fraud_df,
        ip_country_df,
        how='left',
        left_on='ip_int',
        right_on=['lower_int', 'upper_int']
    )
    # Optimize with numpy broadcasting for large datasets
    return merged_df