import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from decision.mlp_model import NeuroDecisionMLP
import joblib
import os

class NeuroDriveDataset(Dataset):
    def __init__(self, X, y_steer, y_brake):
        self.X = torch.FloatTensor(X)
        self.y_steer = torch.FloatTensor(y_steer).unsqueeze(1)
        self.y_brake = torch.FloatTensor(y_brake).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y_steer[idx], self.y_brake[idx]

def train_pipeline(data_path="dataset/hybrid_features.csv", epochs=30, batch_size=32):
    print("Initializing NeuroDrive feature-driven training pipeline...")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    
    # 2. Categorical Encoding (closest_object_type)
    type_map = {"car": 0, "pedestrian": 1, "bike": 2, "truck": 3, "none": 4}
    df['closest_object_type'] = df['closest_object_type'].map(type_map)
    
    X = df[['distance_to_object', 'relative_velocity', 'lane_offset', 
            'lane_curvature', 'num_objects', 'closest_object_type']].values
    
    y_steer = df['steering_angle'].values
    y_brake = df['brake'].values
    
    # 3. 80/20 Train-Test Split
    X_train, X_test, ys_train, ys_test, yb_train, yb_test = train_test_split(
        X, y_steer, y_brake, test_size=0.2, random_state=42
    )
    
    # 4. Fit StandardScaler strictly on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for consistent inference later
    os.makedirs('weights', exist_ok=True)
    joblib.dump(scaler, 'weights/feature_scaler.pkl')
    
    # 5. Dataloaders
    train_dataset = NeuroDriveDataset(X_train_scaled, ys_train, yb_train)
    test_dataset = NeuroDriveDataset(X_test_scaled, ys_test, yb_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 6. Initialize Model and Optimizers
    model = NeuroDecisionMLP(input_features=6)
    
    steer_criterion = nn.MSELoss()
    brake_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Commencing Training ({epochs} epochs, {len(X_train)} train, {len(X_test)} validation)...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_ys, batch_yb in train_loader:
            optimizer.zero_grad()
            
            p_steer, p_brake = model(batch_x)
            
            # Combine losses
            loss_s = steer_criterion(p_steer, batch_ys)
            loss_b = brake_criterion(p_brake, batch_yb)
            loss = loss_s + loss_b
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for batch_x, batch_ys, batch_yb in test_loader:
                 p_steer, p_brake = model(batch_x)
                 loss_s = steer_criterion(p_steer, batch_ys)
                 loss_b = brake_criterion(p_brake, batch_yb)
                 val_loss += (loss_s + loss_b).item()
                 
        t_l = train_loss/len(train_loader)
        v_l = val_loss/len(test_loader)
        if (epoch+1) % 10 == 0 or epoch == 0:
             print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {t_l:.4f} | Val Loss: {v_l:.4f}")
             
    # Save Model
    torch.save(model.state_dict(), 'weights/neurodrive_mlp.pth')
    print("Training complete. Weights and Scaler saved.")
    
if __name__ == "__main__":
    train_pipeline()
