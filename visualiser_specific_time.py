import polars as pl
import os
from utils import *

time_to_plot = "2022-07-30 05:00:00"
ym = int(time_to_plot[:4]+time_to_plot[5:7])
nextcloud_root_dir = os.path.expanduser('~/Nextcloud2/Beegroup/data/CR_BCN_meteo')
plots_dir = f'{nextcloud_root_dir}/Plots_validation'
os.makedirs(plots_dir, exist_ok=True)
predictions_folder = f'{nextcloud_root_dir}/Historical_ERA5Land/Predictions'
geometries_folder = f'{nextcloud_root_dir}/General_Data'
y2 = int(str(ym)[0:4])
m2 = int(str(ym)[4:6])
if (m2-1) == 0:
    y2 = y2-1
    m2 = 12
ym2 = int(f'{y2}{m2:02}')
if os.path.exists(f"{predictions_folder}/prediction_{ym}.parquet"):
    df = pl.read_parquet(f"{predictions_folder}/prediction_{ym}.parquet")
else:
    raise Exception("No prediction file exists")
if os.path.exists(f"{predictions_folder}/prediction_{ym2}.parquet"):
    df = pl.concat([pl.read_parquet(f"{predictions_folder}/prediction_{ym2}.parquet"), df])
df = df.with_columns([
    pl.col("weatherStation").str.split("_").list.get(0).alias("latitude"),
    pl.col("weatherStation").str.split("_").list.get(1).alias("longitude")
])
plot_air_temperature_raster(df.drop("weatherStation"), time_to_plot, max_area=0.000001,
                            pdf_path=f"{plots_dir}/results_temperature_{time_to_plot}_new.pdf")