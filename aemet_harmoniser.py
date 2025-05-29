import os
import json
import pandas as pd

# Directories
wd = "/home/gmor/Nextcloud2/Beegroup/data/AEMET"
wd_clean = "/home/gmor/Nextcloud2/Beegroup/data/AEMET/clean"

os.makedirs(wd, exist_ok=True)
os.makedirs(wd_clean, exist_ok=True)

# Column renaming map
COLUMN_MAP = {
    "idema": "station_id",
    "lon": "longitude",
    "lat": "latitude",
    "alt": "altitude",
    "ubi": "location",
    "fint": "datetime",
    "prec": "precipitation_mm",
    "pacutp": "disdrometer_precip_mm",
    "pliqtp": "liquid_precip_mm",
    "psolt": "solid_precip_mm",
    "vmax": "wind_max_speed_ms",
    "vv": "wind_avg_speed_ms",
    "vmaxu": "wind_max_speed_ultrasonic_ms",
    "vvu": "wind_avg_speed_ultrasonic_ms",
    "dv": "wind_direction_deg",
    "dvu": "wind_direction_ultrasonic_deg",
    "dmax": "wind_max_direction_deg",
    "dmaxu": "wind_max_direction_ultrasonic_deg",
    "stdvv": "wind_speed_stddev_ms",
    "stddv": "wind_direction_stddev_deg",
    "stdvvu": "wind_speed_stddev_ultrasonic_ms",
    "stddvu": "wind_direction_stddev_ultrasonic_deg",
    "hr": "relative_humidity_pct",
    "inso": "sunshine_duration_hr",
    "pres": "pressure_station_hpa",
    "pres_nmar": "pressure_sealevel_hpa",
    "ts": "soil_temp_c",
    "tss20cm": "subsoil_temp_20cm_c",
    "tss5cm": "subsoil_temp_5cm_c",
    "ta": "air_temp_c",
    "tpr": "dew_point_temp_c",
    "tamin": "air_temp_min_c",
    "tamax": "air_temp_max_c",
    "vis": "visibility_km",
    "geo700": "geopotential_height_700",
    "geo850": "geopotential_height_850",
    "geo925": "geopotential_height_925",
    "rviento": "wind_run_hm",
    "nieve": "snow_depth_cm"
}

# Get .json files to process
json_files = [f for f in os.listdir(wd) if f.endswith(".json") and os.path.isfile(os.path.join(wd, f))]

# Exit early if no raw files
if not json_files:
    print("✅ No new files to clean. Exiting.")
    exit(0)

# Process files
for filename in json_files:
    path = os.path.join(wd, filename)

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ Skipping malformed JSON: {filename}")
        continue

    data = content.get("data")
    if not data:
        print(f"⚠️ No 'data' in file: {filename}")
        continue

    df_new = pd.DataFrame(data)

    if df_new.empty:
        print(f"⚠️ No rows in file: {filename}")
        continue

    df_new = df_new.rename(columns=COLUMN_MAP)
    df_new["datetime"] = pd.to_datetime(df_new["datetime"], errors="coerce")
    df_new = df_new.dropna(subset=["station_id", "datetime"])
    df_new = df_new.sort_values("datetime")

    # Output Parquet path
    parquet_path = os.path.join(wd_clean, f"{df_new['station_id'].iloc[0]}.parquet")

    # Load existing cleaned data if present
    if os.path.exists(parquet_path):
        df_existing = pd.read_parquet(parquet_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Deduplicate and sort
    df_combined = df_combined.drop_duplicates(subset=["station_id", "datetime"], keep="last")
    df_combined = df_combined.sort_values("datetime")

    # Save cleaned data
    df_combined.to_parquet(parquet_path, index=False)

    # Delete processed raw file
    os.remove(path)
    print(f"✅ Processed and cleaned: {filename}")

print("🎉 All available files processed.")
