import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.train_model import preprocess_data, MLP

def evaluate_model(model_path, file_path):
    features, labels = preprocess_data(file_path)
    features, labels = torch.tensor(features, dtype=torch.float32), torch.tensor(labels.values, dtype=torch.float32)
    
    # Load model
    input_size = features.shape[1]
    model = MLP(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        predictions = model(features).squeeze()
        predictions = (predictions > 0.5).int()
    
    # Print evaluation metrics
    print(classification_report(labels, predictions))
    print(confusion_matrix(labels, predictions))

if __name__ == "__main__":
    evaluate_model('model/intrusion_model.pth', 'data/nsl-kdd.csv')
