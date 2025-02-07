
import numpy as np

def ip_to_int(ip):
    """Convert IP address string to integer."""
    octets = list(map(int, ip.split('.')))
    return (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]

def extract_time_features(df, time_col):
    """Extract hour_of_day and day_of_week from timestamp."""
    df[f'{time_col}_hour'] = df[time_col].dt.hour
    df[f'{time_col}_dow'] = df[time_col].dt.dayofweek
    return df