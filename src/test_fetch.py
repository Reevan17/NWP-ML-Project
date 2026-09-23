import sys
import time
import traceback

print("="*60)
print("🔍 RUNNING DIAGNOSTIC TEST FOR GFS & ERA5 ACCESS")
print("="*60)

# Test 1: Check required packages
print("\n[Step 1] Checking Python Packages...")
required_pkgs = ['torch', 'xarray', 'zarr', 'gcsfs', 'herbie', 'scipy', 'pandas']
missing = []
for pkg in required_pkgs:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg} is installed")
    except ImportError:
        print(f"  ❌ {pkg} is MISSING")
        missing.append(pkg)

# Check GRIB backend
try:
    import cfgrib
    print("  ✅ cfgrib (GRIB engine) is installed")
except ImportError:
    print("  ⚠️ cfgrib is NOT installed (Run: pip install cfgrib)")

# Test 2: Test NOAA GFS access via Herbie
print("\n[Step 2] Testing NOAA GFS Fetch (1 sample: 2023-01-01 00:00 UTC)...")
t0 = time.time()
try:
    from herbie import Herbie
    H = Herbie("2023-01-01 00:00", model="gfs", product="pgrb2.0p25", fxx=0)
    print(f"  • Herbie GFS index found: {H.grib}")
    
    # Try reading subset
    ds = H.xarray(":TMP:500 mb")
    print(f"  ✅ GFS Temperature loaded successfully! Shape: {ds['t'].shape} (Took {time.time()-t0:.1f}s)")
except Exception as e:
    print(f"  ❌ GFS Fetch Error: {e}")
    traceback.print_exc()

# Test 3: Test Google Cloud ERA5 Zarr access
print("\n[Step 3] Testing Google Cloud ERA5 Access...")
t0 = time.time()
try:
    import gcsfs
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token='anon')
    store = fs.get_mapper('gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
    ds_era = xr.open_zarr(store, chunks=None)
    val = ds_era['temperature'].sel(time="2023-01-01T00:00:00", level=500, latitude=slice(40, 5), longitude=slice(65, 100)).values
    print(f"  ✅ ERA5 Temperature slice loaded successfully! Shape: {val.shape} (Took {time.time()-t0:.1f}s)")
except Exception as e:
    print(f"  ❌ ERA5 Fetch Error: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("Diagnostic Complete!")
print("="*60)
