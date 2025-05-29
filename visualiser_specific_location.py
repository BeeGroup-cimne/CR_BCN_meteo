import os
import polars as pl
import pandas as pd
import numpy as np
from utils import plot_time_series_bundle
from heat_kpis import interpolate_heat_index

locations = [
    {"name": "Besòs Avinguda Prim", "coords": (41.425275, 2.191177), "nearest": None},
    {"name": "Raval estació meteocat", "coords": (41.3839, 2.16775), "nearest": None},
    {"name": "Balmes amb Provença", "coords": (41.392431, 2.158492), "nearest": None},
    {"name": "Via Augusta amb Muntaner", "coords": (41.398897, 2.143193), "nearest": None},
    {"name": "Parc del Carmel", "coords": (41.418678, 2.154480), "nearest": None},
    {"name": "Parc de la Guineueta", "coords": (41.441153, 2.172296), "nearest": None},
    {"name": "Nus de la Trinitat", "coords": (41.448974, 2.194748), "nearest": None},
    {"name": "La Verneda", "coords": (41.426498,2.201928), "nearest": None},
    {"name": "Sagrada Família", "coords": (41.403418,2.173621), "nearest": None},
    {"name": "Estadi Lluís Companys", "coords": (41.364725, 2.157152), "nearest": None},
    {"name": "Observatori Fabra", "coords": (41.4184332729963, 2.124204684660523), "nearest": None}

] # https://www.latlong.net/

nextcloud_root_dir = os.path.expanduser('~/Nextcloud2/Beegroup/data/CR_BCN_meteo')
predictions_dir = f'{nextcloud_root_dir}/Historical_ERA5Land/Predictions'
plots_dir = f'{nextcloud_root_dir}/Plots_validation'
os.makedirs(plots_dir, exist_ok=True)

files = os.listdir(predictions_dir)
files_by_ym = sorted(
    [f for f in files if f.endswith(".parquet")],
    key=lambda x: int(x.replace("prediction_", "").replace(".parquet", ""))
)

df = None
for file in files_by_ym:
    print(file)
    df_ = pl.read_parquet(f"{predictions_dir}/{file}")
    for idx, location in enumerate(locations):
        if location["nearest"] is None:
            ws_unique = df_["weatherStation"].unique()
            hr_ws = pd.DataFrame({
                "lat": ws_unique.str.split("_").list.get(0),
                "lon": ws_unique.str.split("_").list.get(1)
            })
            hr_ws = hr_ws.astype(float)
            hr_ws["dist"] = np.sqrt((hr_ws["lat"] - location["coords"][0]) ** 2 +
                                    (hr_ws["lon"] - location["coords"][1]) ** 2)
            closest = hr_ws.loc[hr_ws["dist"].idxmin()]
            locations[idx]["nearest"] = (closest["lat"], closest["lon"])
    df_ = df_.filter(pl.col("weatherStation").is_in(
        [f"{loc['nearest'][0]}_{loc['nearest'][1]}" for loc in locations]
    ))
    df_ = df_.with_columns([
        pl.col("weatherStation").alias("locationName")
    ])
    df_ = df_.with_columns([
        pl.col("locationName").replace({
            f"{loc['nearest'][0]}_{loc['nearest'][1]}": loc["name"] for loc in locations
        })
    ])
    if df is None:
        df = df_
    else:
        df = pl.concat([df, df_])

tdb = df["airTemperature"].to_list()
rh = [r * 100 for r in df["relativeHumidity"].to_list()]

hi_values = interpolate_heat_index(tdb, rh)

df = df.with_columns(
    pl.Series("heatIndex", hi_values)
)

df = df.with_columns([
    pl.col("time").dt.hour().alias("hour"),
    pl.col("time").dt.date().alias("date")
])

# Define masks for night (0–8, 21–23) and daylight (9–20)
night_mask = (pl.col("hour") <= 8) | (pl.col("hour") >= 21)
day_mask = (pl.col("hour") >= 9) & (pl.col("hour") <= 20)

# Nighttime stats per day and location
night_stats = (
    df.filter(night_mask)
    .group_by(["date", "locationName"])
    .agg([
        pl.col("heatIndex").min().alias("night_temp_min"),
        pl.col("heatIndex").mean().alias("night_temp_mean"),
        pl.col("heatIndex").max().alias("night_temp_max")
    ])
)

# Daylight stats per day and location
day_stats = (
    df.filter(day_mask)
    .group_by(["date", "locationName"])
    .agg([
        pl.col("heatIndex").min().alias("day_temp_min"),
        pl.col("heatIndex").mean().alias("day_temp_mean"),
        pl.col("heatIndex").max().alias("day_temp_max")
    ])
)

# Merge both on date and locationName
daily_temp_stats = night_stats.join(day_stats, on=["date", "locationName"], how="full").sort(["locationName", "date"])
daily_temp_stats = daily_temp_stats.with_columns(
    pl.col("date").alias("time")
)

plot_time_series_bundle(
    df=df,
    group_by_column="locationName",
    title="Weather Station",
    variables=["airTemperature","heatIndex"],
    output_file=f"{plots_dir}/weather_timeseries.html",
    y_limits_axis=(0,40)
)

plot_time_series_bundle(
    df=daily_temp_stats,
    group_by_column="locationName",
    title="Weather station",
    variables=["day_temp_max","day_temp_min", "night_temp_min", "night_temp_max"],
    output_file=f"{plots_dir}/weather_daily_timeseries.html",
    limits={
        "day_temp_max": {"value": 33.0, "condition": "above"}
    },
    y_limits_axis=(0,40)
)