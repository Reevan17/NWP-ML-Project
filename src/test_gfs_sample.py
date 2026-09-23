import time
import numpy as np
from herbie import Herbie
from scipy.ndimage import zoom

LAT_SLICE = slice(40.0, 5.0)
LON_SLICE = slice(65.0, 100.0)
TARGET_SHAPE = (128, 128)

def resize_grid(arr, target_shape=TARGET_SHAPE):
    zoom_factors = (target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1])
    return zoom(arr, zoom_factors, order=1)

print("Testing end-to-end sample build for 2023-01-01 00:00 UTC...")
t0 = time.time()

H = Herbie("2023-01-01 00:00", model="gfs", product="pgrb2.0p25", fxx=0, verbose=False)
ds_t  = H.xarray(":TMP:500 mb", verbose=False).sel(latitude=LAT_SLICE, longitude=LON_SLICE)
ds_gh = H.xarray(":HGT:500 mb", verbose=False).sel(latitude=LAT_SLICE, longitude=LON_SLICE)
ds_u  = H.xarray(":UGRD:500 mb", verbose=False).sel(latitude=LAT_SLICE, longitude=LON_SLICE)
ds_v  = H.xarray(":VGRD:500 mb", verbose=False).sel(latitude=LAT_SLICE, longitude=LON_SLICE)
ds_r  = H.xarray(":RH:500 mb", verbose=False).sel(latitude=LAT_SLICE, longitude=LON_SLICE)

t_gfs  = resize_grid(np.squeeze(ds_t['t'].values))
gh_gfs = resize_grid(np.squeeze(ds_gh['gh'].values))
u_gfs  = resize_grid(np.squeeze(ds_u['u'].values))
v_gfs  = resize_grid(np.squeeze(ds_v['v'].values))
r_gfs  = resize_grid(np.squeeze(ds_r['r'].values))

X = np.stack([t_gfs, gh_gfs, u_gfs, v_gfs, r_gfs], axis=0)
print(f"✅ GFS 5-channel Input X created! Shape: {X.shape} (Took {time.time()-t0:.2f}s)")
