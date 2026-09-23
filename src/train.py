import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Resolve project base directory dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_unet_model.pth")
NORM_STATS_PATH = os.path.join(MODELS_DIR, "norm_stats.npz")

# Local import support
try:
    from model import WeatherUNet
except ImportError:
    from src.model import WeatherUNet

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 50

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple M1/M2/M3 Silicon GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def load_data():
    print(f"Loading datasets on {DEVICE}...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    Y_val   = np.load(os.path.join(DATA_DIR, "Y_val.npy"))

    # Normalization Statistics calculated strictly on the Training set
    mean_X = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std_X  = np.std(X_train, axis=(0, 2, 3), keepdims=True) + 1e-6

    mean_Y = np.mean(Y_train)
    std_Y  = np.std(Y_train) + 1e-6

    np.savez(NORM_STATS_PATH, mean_X=mean_X, std_X=std_X, mean_Y=mean_Y, std_Y=std_Y)

    # Standardize tensors
    X_train_norm = (X_train - mean_X) / std_X
    Y_train_norm = (Y_train - mean_Y) / std_Y
    X_val_norm   = (X_val - mean_X) / std_X
    Y_val_norm   = (Y_val - mean_Y) / std_Y

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), 
                      torch.tensor(Y_train_norm, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32), 
                      torch.tensor(Y_val_norm, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, val_loader, mean_Y, std_Y

def train():
    train_loader, val_loader, mean_Y, std_Y = load_data()
    model = WeatherUNet(in_channels=5, out_channels=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"\n{'Epoch':^7} | {'Train Loss (MSE)':^18} | {'Val Loss (MSE)':^16} | {'Val RMSE (°C)':^15}")
    print("-" * 65)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        val_rmse_celsius = np.sqrt(val_loss) * std_Y

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss: #not printing all lines
            print(f"{epoch:^7d} | {train_loss:^18.5f} | {val_loss:^16.5f} | {val_rmse_celsius:^15.2f} °C")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n✅ Training Finished! Best model saved to '{MODEL_SAVE_PATH}'.")

if __name__ == "__main__":
    train()
