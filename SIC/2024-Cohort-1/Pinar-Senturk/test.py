import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from models.encoder_model import Encoder

def load_sample_ecg(csv_file):
    try:
        df = pd.read_csv(csv_file)
        ecg_signal = df['ecg_signal'].values
        return ecg_signal
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
def main():
    sample_ecg_path = os.path.join(os.getcwd(), 'sample_ecg.csv')
    sample_ecg = load_sample_ecg(sample_ecg_path)

    if sample_ecg is None:
        print("Failed to load sample ECG data.")
        return

    scaler = StandardScaler()
    sample_ecg_normalized = scaler.fit_transform(sample_ecg.reshape(-1, 1)).reshape(-1)
    sample_ecg_tensor = torch.tensor(sample_ecg_normalized, dtype=torch.float32)
    sample_ecg_tensor = sample_ecg_tensor.unsqueeze(0).unsqueeze(0)  

    model_path = os.path.join(os.getcwd(), 'torch_models', 'best_model.pt')

    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found.")
        return

    model = Encoder(enc_space_dim=5) 
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    with torch.no_grad():
        output = model(sample_ecg_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    class_names = {
        0: 'Normal beat',
        1: 'Supraventricular premature beat',
        2: 'Premature ventricular contraction',
        3: 'Fusion of ventricular and normal beat',
        4: 'Unclassifiable beat'
    }

    predicted_class_label = class_names[predicted_class]
    print(f"Predicted ECG class: {predicted_class_label}")

if __name__ == "__main__":
    main()