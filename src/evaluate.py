import os
import torch
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_unet_model.pth")
NORM_STATS_PATH = os.path.join(MODELS_DIR, "norm_stats.npz")

try:
    from model import WeatherUNet
except ImportError:
    from src.model import WeatherUNet

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def add_physics_channels(X):
    N, C, H, W = X.shape
    y_coords = np.linspace(-1, 1, H, dtype=np.float32).reshape(1, 1, H, 1)
    y_coords = np.tile(y_coords, (N, 1, 1, W))
    
    x_coords = np.linspace(-1, 1, W, dtype=np.float32).reshape(1, 1, 1, W)
    x_coords = np.tile(x_coords, (N, 1, H, 1))
    
    gh = X[:, 4:5, :, :]
    gh_anomaly = gh - np.mean(gh, axis=(2, 3), keepdims=True)
    return np.concatenate([X, y_coords, x_coords, gh_anomaly], axis=1)

def evaluate():
    stats = np.load(NORM_STATS_PATH)
    mean_X, std_X = stats['mean_X'], stats['std_X']
    mean_delta, std_delta = float(stats['mean_delta']), float(stats['std_delta'])

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    X_test_8 = add_physics_channels(X_test)
    X_test_norm = torch.tensor((X_test_8 - mean_X) / std_X, dtype=torch.float32).to(DEVICE)

    model = WeatherUNet(in_channels=8, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred_delta_norm = model(X_test_norm).cpu().numpy()

    # Denormalize residual delta and add directly to GFS
    pred_delta = (pred_delta_norm * std_delta) + mean_delta
    gfs_kelvin = X_test[:, 0:1, :, :]
    preds_kelvin = gfs_kelvin + pred_delta
    era_kelvin = Y_test

    # Compute Statistical Metrics (RMSE in °C / K)
    gfs_rmse = np.sqrt(np.mean((gfs_kelvin - era_kelvin)**2))
    ai_rmse  = np.sqrt(np.mean((preds_kelvin - era_kelvin)**2))
    reduction = ((gfs_rmse - ai_rmse) / gfs_rmse) * 100

    print("="*55)
    print(f"🏆 FINAL EVALUATION RESULTS (TEST SET: {len(X_test)} SAMPLES)")
    print("="*55)
    print(f"• Baseline Raw GFS Error: {gfs_rmse:.2f} °C")
    print(f"• AI Corrected Error:     {ai_rmse:.2f} °C")
    print(f"• Accuracy Improvement:   {reduction:.1f} % Error Reduction!")
    print("="*55)

    # Aerodynamic Drag Impact Calculation
    p = 50000.0       # 500 hPa = 50,000 Pa
    R_d = 287.05      # J/(kg*K)
    v = 1000.0        # ~ Mach 3 supersonic booster re-entry (1000 m/s)
    Cd = 0.82         # Booster drag coefficient
    A = 10.5          # Cross-sectional area (m^2, Falcon 9 scale)

    rho_era = p / (R_d * era_kelvin)
    rho_gfs = p / (R_d * gfs_kelvin)
    rho_ai  = p / (R_d * preds_kelvin)

    drag_era = 0.5 * rho_era * (v**2) * Cd * A / 1000.0  # in kN
    drag_gfs = 0.5 * rho_gfs * (v**2) * Cd * A / 1000.0
    drag_ai  = 0.5 * rho_ai  * (v**2) * Cd * A / 1000.0

    drag_error_gfs = np.mean(np.abs(drag_gfs - drag_era))
    drag_error_ai  = np.mean(np.abs(drag_ai - drag_era))
    drag_reduction = ((drag_error_gfs - drag_error_ai) / drag_error_gfs) * 100

    print(f"🚀 Booster Drag Error at 500 hPa (Full Test Set Mean):")
    print(f"   • Raw GFS Drag Discrepancy: {drag_error_gfs:.2f} kN")
    print(f"   • AI Corrected Discrepancy: {drag_error_ai:.2f} kN")
    print(f"   • Drag Accuracy Boost:      {drag_reduction:.1f} % Error Reduction!")
    print("="*55)

    # Plot 4-Panel Verification Figure
    idx = 0
    vmin = min(era_kelvin[idx, 0].min(), gfs_kelvin[idx, 0].min(), preds_kelvin[idx, 0].min()) - 273.15
    vmax = max(era_kelvin[idx, 0].max(), gfs_kelvin[idx, 0].max(), preds_kelvin[idx, 0].max()) - 273.15

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    im0 = axes[0, 0].imshow(gfs_kelvin[idx, 0] - 273.15, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"(A) Raw GFS Forecast ($T_{{500hPa}}$)\nRMSE: {np.sqrt(np.mean((gfs_kelvin[idx]-era_kelvin[idx])**2)):.2f} °C", fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], label="°C")

    im1 = axes[0, 1].imshow(era_kelvin[idx, 0] - 273.15, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("(B) Ground Truth (ECMWF ERA5)\nReference Field", fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], label="°C")

    im2 = axes[1, 0].imshow(preds_kelvin[idx, 0] - 273.15, cmap='jet', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"(C) Stage 7 Physics-CoordConv AI Corrected\nRMSE: {np.sqrt(np.mean((preds_kelvin[idx]-era_kelvin[idx])**2)):.2f} °C", fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 0], label="°C")

    error_diff = np.abs(preds_kelvin[idx, 0] - era_kelvin[idx, 0])
    im3 = axes[1, 1].imshow(error_diff, cmap='inferno', vmin=0, vmax=2.5)
    axes[1, 1].set_title("(D) Remaining Absolute Error ($|T_{{AI}} - T_{{ERA5}}|$\nResidual Discrepancy", fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 1], label="|ΔT| °C")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Stage 7: AI-Driven Atmospheric Bias Correction at 500 hPa\n(South Asia Domain: 5°N–40°N, 65°E–100°E)", fontsize=15, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "rocket_trajectory_weather_correction.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Master Plot saved to '{plot_path}'")

if __name__ == "__main__":
    evaluate()