import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(file_path):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])
    data['service'] = label_encoder.fit_transform(data['service'])
    data['flag'] = label_encoder.fit_transform(data['flag'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    features = data.drop('target', axis=1)
    labels = data['target']
    features = scaler.fit_transform(features)
    
    return features, labels
