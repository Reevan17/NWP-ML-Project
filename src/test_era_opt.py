import time
import gcsfs
import xarray as xr
import numpy as np

print("Testing Optimized ERA5 Zarr loading...")
t0 = time.time()
fs = gcsfs.GCSFileSystem(token='anon')
store = fs.get_mapper('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')

# Open with consolidated metadata or specific cache
ds_era = xr.open_zarr(store, chunks=None, consolidated=True)
print(f"Dataset opened in {time.time()-t0:.2f}s")

# Let's inspect coordinates to find exact integer indices for 40N-5N and 65E-100E
lats = ds_era['latitude'].values
lons = ds_era['longitude'].values
levels = ds_era['level'].values

lat_mask = (lats <= 40.0) & (lats >= 5.0)
lon_mask = (lons >= 65.0) & (lons <= 100.0)

lat_indices = np.where(lat_mask)[0]
lon_indices = np.where(lon_mask)[0]
level_500_idx = int(np.where(levels == 500)[0][0])

lat_min_idx, lat_max_idx = lat_indices[0], lat_indices[-1] + 1
lon_min_idx, lon_max_idx = lon_indices[0], lon_indices[-1] + 1

print(f"Lat slice indices: {lat_min_idx}:{lat_max_idx}")
print(f"Lon slice indices: {lon_min_idx}:{lon_max_idx}")
print(f"500 hPa level index: {level_500_idx}")

# Test fetching slice by isel vs sel
t1 = time.time()
temp_var = ds_era['temperature']
# Slicing using isel
slice_val = temp_var.sel(time="2023-01-01T00:00:00").isel(
    level=level_500_idx,
    latitude=slice(lat_min_idx, lat_max_idx),
    longitude=slice(lon_min_idx, lon_max_idx)
).values
print(f"Slice fetched using isel in {time.time()-t1:.2f}s! Shape: {slice_val.shape}")

t2 = time.time()
slice_val2 = temp_var.sel(time="2023-01-01T12:00:00").isel(
    level=level_500_idx,
    latitude=slice(lat_min_idx, lat_max_idx),
    longitude=slice(lon_min_idx, lon_max_idx)
).values
print(f"Second slice fetched in {time.time()-t2:.2f}s! Shape: {slice_val2.shape}")
