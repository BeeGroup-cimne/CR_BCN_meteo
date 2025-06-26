import polars as pl
import os
from utils import *

time_to_plot = "2025-06-26 14:00:00"
date = f"{time_to_plot[:10]}"
nextcloud_root_dir = os.path.expanduser('~/Nextcloud2/Beegroup/data/CR_BCN_meteo')
plots_dir = f'{nextcloud_root_dir}/Plots_validation'
os.makedirs(plots_dir, exist_ok=True)
predictions_folder = f'{nextcloud_root_dir}/Forecasting_MeteoGalicia/Predictions'
geometries_folder = f'{nextcloud_root_dir}/General_Data'
if os.path.exists(f"{predictions_folder}/prediction_{date}.parquet"):
    df = pl.read_parquet(f"{predictions_folder}/prediction_{date}.parquet")
else:
    raise Exception("No prediction file exists")
df = df.with_columns([
    pl.col("weatherStation").str.split("_").list.get(0).alias("latitude"),
    pl.col("weatherStation").str.split("_").list.get(1).alias("longitude")
])
plot_air_temperature_raster(df.drop("weatherStation"), time_to_plot, max_area=0.000001,
                            pdf_path=f"{plots_dir}/results_temperature_{time_to_plot}_forecasting.pdf")
plot_humidity_raster(df.drop("weatherStation"), time_to_plot, max_area=0.000001,
                            pdf_path=f"{plots_dir}/results_humidity_{time_to_plot}_forecasting.pdf")