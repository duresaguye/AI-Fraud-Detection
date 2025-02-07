
from utils import extract_time_features

def add_transaction_velocity(df, user_col='user_id', time_col='purchase_time'):
    """Calculate number of transactions per user in last 24 hours."""
    df_sorted = df.sort_values(by=time_col)
    df_sorted['transaction_count'] = df_sorted.groupby(user_col)[time_col].transform(
        lambda x: x.rolling('24H', closed='left').count()
    )
    return df_sorted

def engineer_features(df):
    df = extract_time_features(df, 'purchase_time')
    df = add_transaction_velocity(df)
    return df