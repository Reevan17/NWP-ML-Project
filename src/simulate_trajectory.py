import os
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from model import WeatherUNet
except ImportError:
    from src.model import WeatherUNet

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def simulate_reentry_trajectory(t_500_ref, name="Baseline"):
    # Booster parameters (Falcon 9 Class)
    m = 25600.0                # Dry mass + residual landing fuel (kg)
    Cd = 0.82                  # Supersonic drag coefficient
    A = 10.52                  # Cross-sectional area (m^2)
    gamma = np.radians(72.0)   # Flight path angle
    g = 9.81                   # Gravity (m/s^2)
    R_spec = 287.058           # Specific gas constant (J/kg*K)

    z = 30000.0                # Entry altitude (m)
    v = 1450.0                 # Entry velocity (m/s ~ Mach 4.8)
    x = 0.0                    # Downrange distance (m)
    t = 0.0
    dt = 0.05

    t_hist, z_hist, v_hist, x_hist, drag_hist, q_hist = [], [], [], [], [], []

    while z > 0 and v > 0:
        if z > 11000:
            P = 22632.0 * np.exp(-g * (z - 11000) / (R_spec * 216.65))
            T_base = 216.65
        else:
            P = 101325.0 * (1 - 0.0065 * z / 288.15)**5.2561
            T_base = 288.15 - 0.0065 * z

        altitude_weight = np.exp(-((z - 5500.0) / 3500.0)**2)
        T = T_base + (t_500_ref - 253.15) * altitude_weight

        rho = P / (R_spec * T)
        drag_force = 0.5 * rho * (v**2) * Cd * A
        dynamic_pressure = 0.5 * rho * (v**2) / 1000.0  # in kPa

        thrust = 0.0
        if z < 2500.0 and v > 40.0:
            thrust = 320000.0  # Merlin 1D landing burn

        dv_dt = g * np.sin(gamma) - (drag_force + thrust) / m
        dz_dt = - v * np.sin(gamma)
        dx_dt = v * np.cos(gamma)

        z += dz_dt * dt
        v += dv_dt * dt
        v = max(0.0, v)
        x += dx_dt * dt
        t += dt

        t_hist.append(t)
        z_hist.append(max(0, z / 1000.0))
        v_hist.append(v)
        x_hist.append(x / 1000.0)
        drag_hist.append(drag_force / 1000.0)
        q_hist.append(dynamic_pressure)

    return {
        'time': np.array(t_hist),
        'altitude': np.array(z_hist),
        'velocity': np.array(v_hist),
        'downrange': np.array(x_hist),
        'drag': np.array(drag_hist),
        'q': np.array(q_hist),
        'final_downrange': x_hist[-1] * 1000.0
    }

def main():
    print("="*65)
    print("🚀 3-DOF ROCKET RE-ENTRY FLIGHT TRAJECTORY SIMULATION")
    print("="*65)

    stats = np.load(os.path.join(MODELS_DIR, "norm_stats.npz"))
    mean_X, std_X = stats['mean_X'], stats['std_X']
    mean_delta, std_delta = float(stats['mean_delta']), float(stats['std_delta'])

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    X_test_8 = add_physics_channels(X_test)
    X_test_norm = torch.tensor((X_test_8 - mean_X) / std_X, dtype=torch.float32).to(DEVICE)

    model = WeatherUNet(in_channels=8, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_unet_model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()

    with torch.no_grad():
        pred_deltas = (model(X_test_norm).cpu().numpy() * std_delta) + mean_delta

    T_gfs_all = X_test[:, 0:1, :, :]
    T_ai_all  = T_gfs_all + pred_deltas
    T_era_all = Y_test

    # Compute landing errors across the entire test set (129 flights)
    gfs_misses, ai_misses = [], []
    for i in range(len(X_test)):
        t_gfs = float(np.mean(T_gfs_all[i]))
        t_era = float(np.mean(T_era_all[i]))
        t_ai  = float(np.mean(T_ai_all[i]))

        traj_gfs = simulate_reentry_trajectory(t_gfs)
        traj_era = simulate_reentry_trajectory(t_era)
        traj_ai  = simulate_reentry_trajectory(t_ai)

        gfs_misses.append(abs(traj_gfs['final_downrange'] - traj_era['final_downrange']))
        ai_misses.append(abs(traj_ai['final_downrange'] - traj_era['final_downrange']))

    mean_gfs_miss = np.mean(gfs_misses)
    mean_ai_miss  = np.mean(ai_misses)
    fleet_improvement = ((mean_gfs_miss - mean_ai_miss) / mean_gfs_miss) * 100.0

    print(f"\n📊 FULL TEST SET FLEET BENCHMARK ({len(X_test)} MISSIONS):")
    print(f"• Baseline Raw GFS Mean Landing Error:  {mean_gfs_miss:.2f} meters (Off-target miss)")
    print(f"• AI Corrected Mean Landing Error:       {mean_ai_miss:.2f} meters (Pinpoint landing)")
    print(f"• Fleet Landing Accuracy Boost:         {fleet_improvement:.1f} % Error Reduction!")
    print("="*65 + "\n")

    # Select representative mission (Flight 38: typical synoptic bias)
    demo_idx = 38
    t_gfs_demo = float(np.mean(T_gfs_all[demo_idx]))
    t_era_demo = float(np.mean(T_era_all[demo_idx]))
    t_ai_demo  = float(np.mean(T_ai_all[demo_idx]))

    traj_gfs = simulate_reentry_trajectory(t_gfs_demo, "Raw GFS")
    traj_era = simulate_reentry_trajectory(t_era_demo, "ERA5 Truth")
    traj_ai  = simulate_reentry_trajectory(t_ai_demo, "AI Corrected")

    demo_gfs_error = abs(traj_gfs['final_downrange'] - traj_era['final_downrange'])
    demo_ai_error  = abs(traj_ai['final_downrange']  - traj_era['final_downrange'])

    # Plot Master 4-Panel Verification
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Altitude vs Velocity
    axes[0, 0].plot(traj_gfs['velocity'], traj_gfs['altitude'], 'r--', label='Raw GFS Planned', linewidth=2)
    axes[0, 0].plot(traj_era['velocity'], traj_era['altitude'], 'k-', label='ERA5 Ground Truth', linewidth=2.5)
    axes[0, 0].plot(traj_ai['velocity'], traj_ai['altitude'], 'g-.', label='AI Corrected Trajectory', linewidth=2)
    axes[0, 0].set_xlabel('Velocity (m/s)', fontweight='bold')
    axes[0, 0].set_ylabel('Altitude (km)', fontweight='bold')
    axes[0, 0].set_title('1. Re-entry Velocity Deceleration Profile', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, linestyle=':')
    axes[0, 0].legend()

    # 2. Drag Load vs Altitude
    axes[0, 1].plot(traj_gfs['drag'], traj_gfs['altitude'], 'r--', label='Raw GFS Drag Load', linewidth=2)
    axes[0, 1].plot(traj_era['drag'], traj_era['altitude'], 'k-', label='ERA5 Actual Drag Load', linewidth=2.5)
    axes[0, 1].plot(traj_ai['drag'], traj_ai['altitude'], 'g-.', label='AI Corrected Drag Load', linewidth=2)
    axes[0, 1].set_xlabel('Aerodynamic Drag Force (kN)', fontweight='bold')
    axes[0, 1].set_ylabel('Altitude (km)', fontweight='bold')
    axes[0, 1].set_title('2. Aerodynamic Drag Force at 500 hPa Entry Regime', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, linestyle=':')
    axes[0, 1].legend()

    # 3. Downrange Flight Path
    axes[1, 0].plot(traj_gfs['downrange'], traj_gfs['altitude'], 'r--', label='Raw GFS Flight Path', linewidth=2)
    axes[1, 0].plot(traj_era['downrange'], traj_era['altitude'], 'k-', label='ERA5 Actual Path', linewidth=2.5)
    axes[1, 0].plot(traj_ai['downrange'], traj_ai['altitude'], 'g-.', label='AI Corrected Path', linewidth=2)
    axes[1, 0].set_xlabel('Downrange Distance (km)', fontweight='bold')
    axes[1, 0].set_ylabel('Altitude (km)', fontweight='bold')
    axes[1, 0].set_title('3. Flight Trajectory & Drone Ship Touchdown Corridor', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, linestyle=':')
    axes[1, 0].legend()

    # 4. Fleet Landing Miss-Distance Benchmark
    bars = axes[1, 1].bar(['Raw GFS Fleet Mean', 'AI Corrected Fleet Mean', 'Flight 38 (GFS)', 'Flight 38 (AI)'], 
                          [mean_gfs_miss, mean_ai_miss, demo_gfs_error, demo_ai_error], 
                          color=['#E74C3C', '#2ECC71', '#E67E22', '#27AE60'], width=0.55)
    axes[1, 1].set_ylabel('Touchdown Miss Distance (meters)', fontweight='bold')
    axes[1, 1].set_title(f'4. Landing Dispersion Benchmark ({fleet_improvement:.1f}% Fleet Error Reduction)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', linestyle=':')
    for bar in bars:
        h = bar.get_height()
        axes[1, 1].annotate(f'{h:.1f} m',
                            xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.suptitle("AI-NWP 3-DOF Re-entry Trajectory & Drone Ship Recovery Optimization", fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "rocket_reentry_trajectory_simulation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Master Trajectory Plot saved to '{output_path}'\n")

if __name__ == "__main__":
    main()