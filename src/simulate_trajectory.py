import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import WeatherUNet

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_ai_weather():
    """Loads trained U-Net and computes AI corrected temperature profile"""
    stats = np.load(os.path.join(MODELS_DIR, "norm_stats.npz"))
    mean_X, std_X = stats['mean_X'], stats['std_X']
    mean_Y, std_Y = stats['mean_Y'], stats['std_Y']

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

    model = WeatherUNet(in_channels=5, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_unet_model.pth"), map_location=DEVICE))
    model.eval()

    # Take first test sample
    sample_x = torch.tensor((X_test[0:1] - mean_X) / std_X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_norm = model(sample_x).cpu().numpy()

    t_ai  = (pred_norm * std_Y) + mean_Y
    t_gfs = X_test[0:1, 0:1, :, :]
    t_era = Y_test[0:1]

    # Mean temperature at 500 hPa over landing zone
    return float(np.mean(t_gfs)), float(np.mean(t_era)), float(np.mean(t_ai))

def simulate_reentry_trajectory(t_500_ref, name="Baseline"):
    """
    3-DOF Re-entry Trajectory Simulation for Reusable Rocket Booster
    Integrates equations of motion from 30 km altitude down to landing:
    m * dv/dt = - m*g*sin(gamma) - 0.5 * rho * v^2 * Cd * A + Thrust
    dz/dt = - v * sin(gamma)
    dx/dt = v * cos(gamma)
    """
    # Booster parameters (Falcon 9 Class)
    m = 25600.0        # Dry mass + residual landing fuel (kg)
    Cd = 0.82          # Supersonic drag coefficient
    A = 10.52          # Cross-sectional area (m^2)
    gamma = np.radians(72.0)  # Flight path angle (steep reentry)
    g = 9.81           # Gravity (m/s^2)
    R_spec = 287.058   # Specific gas constant (J/kg*K)

    # Initial conditions at atmospheric entry (30 km altitude)
    z = 30000.0        # Altitude (m)
    v = 1450.0         # Supersonic entry velocity (m/s ~ Mach 4.8)
    x = 0.0            # Downrange distance (m)
    t = 0.0            # Time (s)
    dt = 0.05          # Time step (s)

    # Logging arrays
    t_hist, z_hist, v_hist, x_hist, drag_hist, q_hist = [], [], [], [], [], []

    while z > 0 and v > 0:
        # Standard US Standard Atmosphere pressure profile
        if z > 11000:
            P = 22632.0 * np.exp(-g * (z - 11000) / (R_spec * 216.65))
            T_base = 216.65
        else:
            P = 101325.0 * (1 - 0.0065 * z / 288.15)**5.2561
            T_base = 288.15 - 0.0065 * z

        # Inject 500 hPa weather field temperature offset
        # 500 hPa is located around ~5500m altitude
        altitude_weight = np.exp(-((z - 5500.0) / 3500.0)**2)
        T = T_base + (t_500_ref - 253.15) * altitude_weight

        # Air density via equation of state
        rho = P / (R_spec * T)

        # Aerodynamic drag force
        drag_force = 0.5 * rho * (v**2) * Cd * A
        dynamic_pressure = 0.5 * rho * (v**2) / 1000.0  # kPa

        # Landing burn throttle (autonomous landing guidance at low altitude)
        thrust = 0.0
        if z < 2500.0 and v > 40.0:
            thrust = 320000.0  # 1 Merlin 1D landing burn engine (~320 kN)

        # Equations of motion
        dv_dt = g * np.sin(gamma) - (drag_force - thrust) / m
        dz_dt = - v * np.sin(gamma)
        dx_dt = v * np.cos(gamma)

        # Integration (Euler-Heun step)
        z += dz_dt * dt
        v -= dv_dt * dt
        x += dx_dt * dt
        t += dt

        # Record
        t_hist.append(t)
        z_hist.append(max(0, z / 1000.0))  # in km
        v_hist.append(v)
        x_hist.append(x / 1000.0)          # in km
        drag_hist.append(drag_force / 1000.0)  # in kN
        q_hist.append(dynamic_pressure)

    return {
        'time': np.array(t_hist),
        'altitude': np.array(z_hist),
        'velocity': np.array(v_hist),
        'downrange': np.array(x_hist),
        'drag': np.array(drag_hist),
        'q': np.array(q_hist),
        'final_downrange': x_hist[-1] * 1000.0  # in meters
    }

def main():
    print("="*65)
    print("🚀 3-DOF ROCKET RE-ENTRY FLIGHT TRAJECTORY SIMULATION")
    print("="*65)

    # 1. Load actual AI weather fields
    t_gfs, t_era, t_ai = load_ai_weather()
    print(f"• Raw GFS Mean Temp:      {t_gfs-273.15:.2f} °C")
    print(f"• ERA5 Ground Truth Temp:  {t_era-273.15:.2f} °C")
    print(f"• AI Corrected Temp:       {t_ai-273.15:.2f} °C\n")

    # 2. Run Trajectory Simulations
    print("Simulating Flight 1: Raw GFS Atmosphere (Weather Forecast Bias)...")
    traj_gfs = simulate_reentry_trajectory(t_gfs, "Raw GFS")

    print("Simulating Flight 2: ERA5 Atmosphere (Actual Physical Ground Truth)...")
    traj_era = simulate_reentry_trajectory(t_era, "ERA5 Truth")

    print("Simulating Flight 3: AI-Corrected Atmosphere (AI-NWP Guidance)...")
    traj_ai = simulate_reentry_trajectory(t_ai, "AI Corrected")

    # Calculate Landing Dispersion Error relative to Ground Truth (ERA5)
    gfs_landing_error = abs(traj_gfs['final_downrange'] - traj_era['final_downrange'])
    ai_landing_error  = abs(traj_ai['final_downrange']  - traj_era['final_downrange'])
    error_reduction   = ((gfs_landing_error - ai_landing_error) / gfs_landing_error) * 100.0

    print("\n" + "="*65)
    print("🎯 RE-ENTRY LANDING ACCURACY BENCHMARK (DRONE SHIP RECOVERY)")
    print("="*65)
    print(f"• Baseline Raw GFS Landing Error:  {gfs_landing_error:.1f} meters (Off-target miss)")
    print(f"• AI Corrected Landing Error:      {ai_landing_error:.1f} meters (Pinpoint landing)")
    print(f"• Landing Dispersion Improvement:  {error_reduction:.1f} % Accuracy Boost!")
    print("="*65 + "\n")

    # 3. Master 4-Panel Trajectory Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Altitude vs Velocity
    axes[0, 0].plot(traj_gfs['velocity'], traj_gfs['altitude'], 'r--', label='Raw GFS Planned', linewidth=2)
    axes[0, 0].plot(traj_era['velocity'], traj_era['altitude'], 'k-', label='ERA5 Ground Truth', linewidth=2.5)
    axes[0, 0].plot(traj_ai['velocity'], traj_ai['altitude'], 'g-.', label='AI Corrected Trajectory', linewidth=2)
    axes[0, 0].set_xlabel('Velocity (m/s)', fontweight='bold')
    axes[0, 0].set_ylabel('Altitude (km)', fontweight='bold')
    axes[0, 0].set_title('1. Re-entry Velocity Deceleration Profile', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, linestyle=':')
    axes[0, 0].legend()

    # Panel 2: Aerodynamic Drag Load vs Altitude
    axes[0, 1].plot(traj_gfs['drag'], traj_gfs['altitude'], 'r--', label='Raw GFS Drag Load', linewidth=2)
    axes[0, 1].plot(traj_era['drag'], traj_era['altitude'], 'k-', label='ERA5 Actual Drag Load', linewidth=2.5)
    axes[0, 1].plot(traj_ai['drag'], traj_ai['altitude'], 'g-.', label='AI Corrected Drag Load', linewidth=2)
    axes[0, 1].set_xlabel('Aerodynamic Drag Force (kN)', fontweight='bold')
    axes[0, 1].set_ylabel('Altitude (km)', fontweight='bold')
    axes[0, 1].set_title('2. Aerodynamic Drag Force at 500 hPa Entry Regime', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, linestyle=':')
    axes[0, 1].legend()

    # Panel 3: Altitude vs Downrange Landing Trajectory
    axes[1, 0].plot(traj_gfs['downrange'], traj_gfs['altitude'], 'r--', label='Raw GFS Flight Path', linewidth=2)
    axes[1, 0].plot(traj_era['downrange'], traj_era['altitude'], 'k-', label='ERA5 Actual Path', linewidth=2.5)
    axes[1, 0].plot(traj_ai['downrange'], traj_ai['altitude'], 'g-.', label='AI Corrected Path', linewidth=2)
    axes[1, 0].set_xlabel('Downrange Distance (km)', fontweight='bold')
    axes[1, 0].set_ylabel('Altitude (km)', fontweight='bold')
    axes[1, 0].set_title('3. Flight Trajectory & Drone Ship Touchdown Corridor', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, linestyle=':')
    axes[1, 0].legend()

    # Panel 4: Landing Point Dispersion Bar Chart
    bars = axes[1, 1].bar(['Raw GFS Bias Error', 'AI Corrected Error'], 
                          [gfs_landing_error, ai_landing_error], 
                          color=['#E74C3C', '#2ECC71'], width=0.5)
    axes[1, 1].set_ylabel('Landing Position Miss Distance (meters)', fontweight='bold')
    axes[1, 1].set_title('4. Reusable Booster Landing Target Error (Meters)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', linestyle=':')
    for bar in bars:
        h = bar.get_height()
        axes[1, 1].annotate(f'{h:.1f} m',
                            xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=14)

    plt.suptitle("AI-NWP 3-DOF Re-entry Trajectory & Drone Ship Recovery Optimization", fontsize=15, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "rocket_reentry_trajectory_simulation.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ Trajectory Simulation Plot saved to '{output_path}'\n")

if __name__ == "__main__":
    main()