import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Preprocessing function
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])
    data['service'] = label_encoder.fit_transform(data['service'])
    data['flag'] = label_encoder.fit_transform(data['flag'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    features = data.drop('target', axis=1)
    labels = data['target'].apply(lambda x: 1 if x == 'anomaly' else 0)  # Binary classification
    features = scaler.fit_transform(features)
    
    return features, labels

# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training the model
if __name__ == "__main__":
    # Load and preprocess data
    features, labels = preprocess_data('data/nsl-kdd.csv')
    features, labels = torch.tensor(features, dtype=torch.float32), torch.tensor(labels.values, dtype=torch.float32)
    
    dataset = TensorDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model, loss function, and optimizer
    input_size = features.shape[1]
    model = MLP(input_size)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'model/intrusion_model.pth')
    print("Model training complete and saved!")
