import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data_processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inspect():
    print("="*65)
    print("🔍 DATASET INSPECTION & STATISTICAL SUMMARY")
    print("="*65)
    
    splits = ['train', 'val', 'test']
    for split in splits:
        x_file = os.path.join(DATA_DIR, f"X_{split}.npy")
        y_file = os.path.join(DATA_DIR, f"Y_{split}.npy")
        
        if not os.path.exists(x_file) or not os.path.exists(y_file):
            print(f"⚠️ Missing {split} files at: {x_file}")
            continue
            
        X = np.load(x_file)
        Y = np.load(y_file)
        
        print(f"\n📂 [{split.upper()} SET]")
        print(f"  • Input  X: Shape = {X.shape} | dtype = {X.dtype}")
        print(f"  • Target Y: Shape = {Y.shape} | dtype = {Y.dtype}")
        print(f"  • GFS Temperature (°C): min = {np.min(X[:, 0])-273.15:.1f}°C, max = {np.max(X[:, 0])-273.15:.1f}°C, mean = {np.mean(X[:, 0])-273.15:.1f}°C")
        print(f"  • ERA5 Temperature (°C): min = {np.min(Y)-273.15:.1f}°C, max = {np.max(Y)-273.15:.1f}°C, mean = {np.mean(Y)-273.15:.1f}°C")
        print(f"  • Geopotential Height (m): min = {np.min(X[:, 1]):.0f}m, max = {np.max(X[:, 1]):.0f}m")
        print(f"  • Zonal Wind u (m/s): min = {np.min(X[:, 2]):.1f}, max = {np.max(X[:, 2]):.1f}")
        print(f"  • Meridional Wind v (m/s): min = {np.min(X[:, 3]):.1f}, max = {np.max(X[:, 3]):.1f}")
        print(f"  • Relative Humidity (%): min = {np.min(X[:, 4]):.1f}%, max = {np.max(X[:, 4]):.1f}%")

    print("\n" + "="*65)
    print("🖼️ Generating visual sample inspection plot...")
    
    # Load sample 0 from train set
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    titles = [
        "1. GFS Temperature (°C)",
        "2. GFS Geopotential Height (m)",
        "3. GFS Zonal Wind u (m/s)",
        "4. GFS Meridional Wind v (m/s)",
        "5. GFS Relative Humidity (%)",
        "6. ERA5 Ground Truth Target (°C)"
    ]
    
    data_to_plot = [
        X_train[0, 0] - 273.15,
        X_train[0, 1],
        X_train[0, 2],
        X_train[0, 3],
        X_train[0, 4],
        Y_train[0, 0] - 273.15
    ]
    
    cmaps = ['coolwarm', 'viridis', 'PuOr', 'PuOr', 'YlGnBu', 'coolwarm']
    
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(data_to_plot[i], cmap=cmaps[i])
        ax.set_title(titles[i], fontsize=11, fontweight='bold')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        
    plt.suptitle("AI-NWP: 5-Channel GFS Input Features vs ERA5 Ground Truth (Sample 0)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "dataset_sample_inspection.png")
    plt.savefig(plot_path, dpi=200)
    print(f"✅ Saved inspection plot to '{plot_path}'")

if __name__ == "__main__":
    inspect()
