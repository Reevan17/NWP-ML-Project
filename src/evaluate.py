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

def evaluate():
    stats = np.load(NORM_STATS_PATH)
    mean_X, std_X = stats['mean_X'], stats['std_X']
    mean_delta, std_delta = float(stats['mean_delta']), float(stats['std_delta'])

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    X_test_norm = torch.tensor((X_test - mean_X) / std_X, dtype=torch.float32).to(DEVICE)

    model = WeatherUNet(in_channels=5, out_channels=1).to(DEVICE)
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
    print("🏆 FINAL EVALUATION RESULTS (TEST SET: 129 SAMPLES)")
    print("="*55)
    print(f"• Baseline Raw GFS Error: {gfs_rmse:.2f} °C")
    print(f"• AI Corrected Error:     {ai_rmse:.2f} °C")
    print(f"• Accuracy Improvement:   {reduction:.1f} % Error Reduction!")
    print("="*55)

    # Aerospace Aerodynamic Drag & Density Calculation
    P = 50000.0       # Pascals (500 hPa)
    R_spec = 287.058  # J/(kg*K)
    
    rho_gfs = P / (R_spec * gfs_kelvin)
    rho_era = P / (R_spec * era_kelvin)
    rho_ai  = P / (R_spec * preds_kelvin)

    v = 500.0  # m/s (~Mach 1.6 supersonic reentry speed)
    Cd = 0.8   # Drag coefficient of booster during entry burn / glide
    A = 10.5   # Reference cross-sectional area (m^2) (~Falcon 9 diameter 3.66m)
    
    drag_gfs = 0.5 * rho_gfs * (v**2) * Cd * A / 1000.0  # in kN
    drag_era = 0.5 * rho_era * (v**2) * Cd * A / 1000.0  # in kN
    drag_ai  = 0.5 * rho_ai  * (v**2) * Cd * A / 1000.0  # in kN

    gfs_drag_err = np.mean(np.abs(drag_gfs - drag_era))
    ai_drag_err  = np.mean(np.abs(drag_ai - drag_era))
    drag_reduction = ((gfs_drag_err - ai_drag_err) / gfs_drag_err) * 100.0

    print(f"🚀 Booster Drag Error at 500 hPa (Full Test Set Mean):")
    print(f"   • Raw GFS Drag Discrepancy: {gfs_drag_err:.2f} kN")
    print(f"   • AI Corrected Discrepancy: {ai_drag_err:.2f} kN")
    print(f"   • Drag Accuracy Boost:      {drag_reduction:.1f} % Error Reduction!")
    print("="*55)

    # Master Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    t_min = min(np.min(gfs_kelvin[0,0]-273.15), np.min(era_kelvin[0,0]-273.15))
    t_max = max(np.max(gfs_kelvin[0,0]-273.15), np.max(era_kelvin[0,0]-273.15))

    im1 = axes[0, 0].imshow(gfs_kelvin[0, 0] - 273.15, cmap='coolwarm', vmin=t_min, vmax=t_max)
    axes[0, 0].set_title('1. Raw GFS Input (500 hPa Temp °C)', fontsize=12, fontweight='bold')
    fig.colorbar(im1, ax=axes[0, 0], orientation='horizontal', pad=0.1)

    im2 = axes[0, 1].imshow(era_kelvin[0, 0] - 273.15, cmap='coolwarm', vmin=t_min, vmax=t_max)
    axes[0, 1].set_title('2. ERA5 Ground Truth Target (°C)', fontsize=12, fontweight='bold')
    fig.colorbar(im2, ax=axes[0, 1], orientation='horizontal', pad=0.1)

    im3 = axes[1, 0].imshow(preds_kelvin[0, 0] - 273.15, cmap='coolwarm', vmin=t_min, vmax=t_max)
    axes[1, 0].set_title('3. AI Corrected Temperature Map (°C)', fontsize=12, fontweight='bold')
    fig.colorbar(im3, ax=axes[1, 0], orientation='horizontal', pad=0.1)

    diff = np.abs(gfs_kelvin[0, 0] - era_kelvin[0, 0]) - np.abs(preds_kelvin[0, 0] - era_kelvin[0, 0])
    im4 = axes[1, 1].imshow(diff, cmap='RdYlGn', vmin=-1, vmax=2)
    axes[1, 1].set_title('4. Error Reduction Map (Green = Fixed by AI)', fontsize=12, fontweight='bold')
    fig.colorbar(im4, ax=axes[1, 1], orientation='horizontal', pad=0.1, label='°C Error Eliminated')

    plt.tight_layout()
    output_plot = os.path.join(OUTPUT_DIR, "rocket_trajectory_weather_correction.png")
    plt.savefig(output_plot, dpi=300)
    print(f"✅ Master Plot saved to '{output_plot}'")

if __name__ == "__main__":
    evaluate()