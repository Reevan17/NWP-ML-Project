# 🚀 AI-Driven Atmospheric Field Correction for Aerospace Trajectory & Rocket Recovery

An end-to-end Deep Learning pipeline using a **2D U-Net Convolutional Neural Network** to correct systematic Numerical Weather Prediction (NWP) biases in **NOAA GFS** against **ECMWF ERA5** reanalysis at $500\text{ hPa}$ (~5.5 km altitude) over South Asia for supersonic rocket booster landing recovery (SpaceX Falcon 9 / ISRO RLV).

---

## 📌 Project Overview
* **Domain**: South Asia / Rocket Flight Corridor ($5^\circ\text{N}–40^\circ\text{N}, 65^\circ\text{E}–100^\circ\text{E}$)
* **Pressure Level**: $500\text{ hPa}$ (Middle Troposphere Steering Layer)
* **Input Features ($X$) [5 Channels]**: Temperature ($T$), Geopotential Height ($gh$), Zonal Wind ($u$), Meridional Wind ($v$), Relative Humidity ($r$) from NOAA GFS.
* **Target ($Y$) [1 Channel]**: High-fidelity Temperature ($T$) from ECMWF ERA5 Ground Truth.
* **Deep Learning Architecture**: 2D U-Net CNN with Double Convolution blocks and Skip Connections.
* **Aerospace Impact**: Eliminates atmospheric density ($\rho = \frac{P}{R_{\text{spec}}T}$) and aerodynamic drag force ($F_{\text{drag}} = \frac{1}{2}\rho v^2 C_d A$) errors during supersonic booster descent.

---

## 🛠️ Project Structure
```text
AI-NWP-Project/
├── app.py                   # Interactive Streamlit Web UI Dashboard
├── requirements.txt         # Project dependencies
├── .gitignore               # Ignored heavy data/model files
├── data_processed/          # Preprocessed .npy tensors (Train, Val, Test)
├── models/                  # Saved weights (best_unet_model.pth, norm_stats.npz)
├── outputs/                 # Evaluation & trajectory plots
└── src/
    ├── prepare_dataset.py   # Multi-threaded NOAA AWS S3 GFS + ERA5 data collector
    ├── inspect_dataset.py   # Statistical validation & 6-panel dataset visualizer
    ├── model.py             # 2D Weather U-Net PyTorch architecture
    ├── train.py             # Training loop with Apple Silicon MPS / CUDA support
    ├── evaluate.py          # Baseline vs AI evaluation & 4-panel master plot
    └── simulate_trajectory.py # 3-DOF supersonic rocket booster flight simulator