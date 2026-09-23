import time
import gcsfs
import xarray as xr
import numpy as np

print("Testing Monthly Batch Extraction on Google Cloud ERA5 Zarr...")
t0 = time.time()
fs = gcsfs.GCSFileSystem(token='anon')
store = fs.get_mapper('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
ds_era = xr.open_zarr(store, chunks=None, consolidated=True)

# Select all timestamps for January 2023 at 00 and 12 UTC for our region at 500 hPa in ONE single call!
times = [f"2023-01-{day:02d}T{hour:02d}:00:00" for day in range(1, 32) for hour in [0, 12]]
print(f"Querying batch of {len(times)} timestamps for Jan 2023 in one request...")

t1 = time.time()
jan_slice = ds_era['temperature'].sel(
    time=times,
    level=500,
    latitude=slice(40.0, 5.0),
    longitude=slice(65.0, 100.0)
).values
print(f"✅ Downloaded all {len(times)} ERA5 samples for entire month in {time.time()-t1:.2f} seconds! Shape: {jan_slice.shape}")
