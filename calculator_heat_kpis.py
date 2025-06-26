import os
import polars as pl
from utils import plot_static_features_maps
from heat_kpis import interpolate_heat_index

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

    tdb = df_["airTemperature"].to_list()
    rh = [r * 100 for r in df_["relativeHumidity"].to_list()]

    hi_values = interpolate_heat_index(tdb, rh)

    df_ = df_.with_columns(
        pl.Series("heatIndex", hi_values)
    )

    df_ = df_.with_columns([
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.date().alias("date")
    ])

    # Define masks for night (0–8, 21–23) and daylight (9–20)
    night_mask = (pl.col("hour") <= 8) | (pl.col("hour") >= 21)
    day_mask = (pl.col("hour") >= 9) & (pl.col("hour") <= 20)

    # Nighttime stats per day and location
    night_stats = (
        df_.filter(night_mask)
        .group_by(["date", "weatherStation"])
        .agg([
            pl.col("heatIndex").min().alias("night_temp_min"),
            pl.col("heatIndex").mean().alias("night_temp_mean"),
            pl.col("heatIndex").max().alias("night_temp_max")
        ])
    )

    # Daylight stats per day and location
    day_stats = (
        df_.filter(day_mask)
        .group_by(["date", "weatherStation"])
        .agg([
            pl.col("heatIndex").min().alias("day_temp_min"),
            pl.col("heatIndex").mean().alias("day_temp_mean"),
            pl.col("heatIndex").max().alias("day_temp_max")
        ])
    )

    # Merge both on date and locationName
    daily_temp_stats = night_stats.join(day_stats, on=["date", "weatherStation"], how="full").sort(
        ["weatherStation", "date"])
    daily_temp_stats = daily_temp_stats.with_columns(
        pl.col("date").alias("time")
    )

    if df is None:
        df = daily_temp_stats
    else:
        df = pl.concat([df, daily_temp_stats])

# Drop duplicate/unnecessary columns
df_cleaned = df.drop(["date", "date_right", "weatherStation_right"])

# Define the threshold
threshold = 33

# Apply transformation
result = (
    df_cleaned
    .sort("time")
    .group_by("weatherStation")
    .agg([
        pl.col("time"),
        pl.col("day_temp_max").rolling_min(window_size=3, min_periods=1).alias("rolling_min_day_temp_max"),
        pl.col("night_temp_min"),
        pl.col("day_temp_max"),
        pl.col("day_temp_mean")
    ])
    .explode(["time", "rolling_min_day_temp_max", "night_temp_min", "day_temp_max", "day_temp_mean"])  # to flatten the nested structure
    .with_columns([
        (pl.col("rolling_min_day_temp_max") > threshold).alias("heatWave"),
        (pl.col("night_temp_min") > 20).alias("tropicalNight"),
        (pl.col("night_temp_min") > 25).alias("torridNight"),
        (pl.col("night_temp_min") > 30).alias("infernalNight"),
        (pl.col("day_temp_max") < 27).alias("safeMaxTemperature"),
        ((pl.col("day_temp_max") >= 27) & (pl.col("day_temp_max") < 33)).alias("cautionMaxTemperature"),
        ((pl.col("day_temp_max") >= 33) & (pl.col("day_temp_max") < 41)).alias("extremeCautionMaxTemperature"),
        ((pl.col("day_temp_max") >= 41) & (pl.col("day_temp_max") < 52)).alias("hazardousMaxTemperature"),
        ((pl.col("day_temp_max") >= 52) & (pl.col("day_temp_max") < 92)).alias("extremeHazardousMaxTemperature"),
        (pl.col("day_temp_max") >= 92).alias("beyondHumanLimitMaxTemperature"),
        (pl.col("day_temp_mean") < 27).alias("safeAverageTemperature"),
        ((pl.col("day_temp_mean") >= 27) & (pl.col("day_temp_mean") < 33)).alias("cautionAverageTemperature"),
        ((pl.col("day_temp_mean") >= 33) & (pl.col("day_temp_mean") < 41)).alias("extremeCautionAverageTemperature"),
        ((pl.col("day_temp_mean") >= 41) & (pl.col("day_temp_mean") < 52)).alias("hazardousAverageTemperature"),
        ((pl.col("day_temp_mean") >= 52) & (pl.col("day_temp_mean") < 92)).alias("extremeHazardousAverageTemperature"),
        (pl.col("day_temp_mean") >= 92).alias("beyondHumanLimitAverageTemperature")
    ])
)


result_by_year = (
    result
    .with_columns([
        pl.col("time").dt.year().alias("year")
    ])
    .group_by(["weatherStation", "year"])
    .agg([
        pl.col("heatWave").sum(),
        pl.col("tropicalNight").sum(),
        pl.col("torridNight").sum(),
        pl.col("infernalNight").sum(),
        pl.col("safeMaxTemperature").sum(),
        pl.col("cautionMaxTemperature").sum(),
        pl.col("extremeCautionMaxTemperature").sum(),
        pl.col("hazardousMaxTemperature").sum(),
        pl.col("extremeHazardousMaxTemperature").sum(),
        pl.col("beyondHumanLimitMaxTemperature").sum(),
        pl.col("safeAverageTemperature").sum(),
        pl.col("cautionAverageTemperature").sum(),
        pl.col("extremeCautionAverageTemperature").sum(),
        pl.col("hazardousAverageTemperature").sum(),
        pl.col("extremeHazardousAverageTemperature").sum(),
        pl.col("beyondHumanLimitAverageTemperature").sum()
    ])
    .sort(["weatherStation", "year"])
)
for y in result_by_year["year"].unique():
    result_one_year = result_by_year.filter(pl.col("year")==y).drop(["year"]).to_pandas().set_index("weatherStation")
    plot_static_features_maps(result_one_year, crs="EPSG:4326", cmap="viridis", markersize=1, output_pdf=f"{plots_dir}/heat_indexes_{y}.pdf")
