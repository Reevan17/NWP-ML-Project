import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_unet_model.pth")
NORM_STATS_PATH = os.path.join(MODELS_DIR, "norm_stats.npz")

try:
    from model import WeatherUNet
except ImportError:
    from src.model import WeatherUNet

BATCH_SIZE = 16
LEARNING_RATE = 6.0e-4
EPOCHS = 100

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class SpatialGradientLoss(nn.Module):
    def __init__(self):
        super(SpatialGradientLoss, self).__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        grad_pred_x = nn.functional.conv2d(pred, self.sobel_x, padding=1)
        grad_pred_y = nn.functional.conv2d(pred, self.sobel_y, padding=1)
        grad_tgt_x  = nn.functional.conv2d(target, self.sobel_x, padding=1)
        grad_tgt_y  = nn.functional.conv2d(target, self.sobel_y, padding=1)
        return torch.mean(torch.abs(grad_pred_x - grad_tgt_x) + torch.abs(grad_pred_y - grad_tgt_y))

class CombinedWeatherLoss(nn.Module):
    def __init__(self):
        super(CombinedWeatherLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.grad_loss = SpatialGradientLoss()

    def forward(self, pred, target):
        return self.mse(pred, target) + 0.6 * self.l1(pred, target) + 0.1 * self.grad_loss(pred, target)

def load_data():
    print(f"Loading datasets on {DEVICE}...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    Y_val   = np.load(os.path.join(DATA_DIR, "Y_val.npy"))

    delta_train = Y_train - X_train[:, 0:1, :, :]
    delta_val   = Y_val   - X_val[:, 0:1, :, :]

    mean_X = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std_X  = np.std(X_train, axis=(0, 2, 3), keepdims=True) + 1e-6

    mean_delta = np.mean(delta_train)
    std_delta  = np.std(delta_train) + 1e-6

    np.savez(NORM_STATS_PATH, mean_X=mean_X, std_X=std_X, mean_delta=mean_delta, std_delta=std_delta)

    X_train_norm = (X_train - mean_X) / std_X
    delta_train_norm = (delta_train - mean_delta) / std_delta

    X_val_norm = (X_val - mean_X) / std_X
    delta_val_norm = (delta_val - mean_delta) / std_delta

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), 
                      torch.tensor(delta_train_norm, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_norm, dtype=torch.float32), 
                      torch.tensor(delta_val_norm, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, val_loader, mean_delta, std_delta

def train():
    train_loader, val_loader, mean_delta, std_delta = load_data()
    model = WeatherUNet(in_channels=5, out_channels=1).to(DEVICE)
    
    criterion = CombinedWeatherLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    print(f"\n{'Epoch':^7} | {'Train Loss':^18} | {'Val Loss':^16} | {'Val RMSE (°C)':^15}")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_mse  += torch.mean((outputs - targets)**2).item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_mse  /= len(val_loader.dataset)
        scheduler.step()

        val_rmse_celsius = np.sqrt(val_mse) * float(std_delta)

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss:
            print(f"{epoch:^7d} | {train_loss:^18.5f} | {val_loss:^16.5f} | {val_rmse_celsius:^15.2f} °C")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\n✅ High-Precision Training Finished! Best model saved to '{MODEL_SAVE_PATH}'.")

if __name__ == "__main__":
    train()