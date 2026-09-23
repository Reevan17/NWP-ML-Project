import urllib.request
import json
import time
import numpy as np

print("Testing Open-Meteo ERA5 Reanalysis API (Global 0.25° ERA5 at 500 hPa)...")
t0 = time.time()

# Let's test querying 500 hPa temperature for South Asia bounding box or grid points
url = "https://archive-api.open-meteo.com/v1/archive?latitude=20.0&longitude=80.0&start_date=2023-01-01&end_date=2023-01-31&hourly=temperature_500hPa"

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=10) as response:
    data = json.loads(response.read().decode())
    temps = data['hourly']['temperature_500hPa']
    print(f"✅ Fetched {len(temps)} ERA5 500hPa hourly temperatures in {time.time()-t0:.2f} seconds!")
