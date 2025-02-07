from sklearn.preprocessing import StandardScaler, OneHotEncoder

def normalize_features(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def encode_categorical(df, columns):
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded = encoder.fit_transform(df[columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))
    return pd.concat([df.drop(columns=columns), encoded_df], axis=1)