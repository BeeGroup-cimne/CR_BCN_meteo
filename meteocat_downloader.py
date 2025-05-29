import os
import meteocat_utils

wd = "/home/gmor/Nextcloud2/Beegroup/data/Meteocat"
lat_range = [41.2, 41.6]  # [Garraf, Cardedeu]
lon_range = [1.9, 2.4]
os.makedirs(wd, exist_ok=True)
stations = meteocat_utils.get_complete_socrata_dataset(
    url="https://analisi.transparenciacatalunya.cat/resource/yqwd-vj5e.csv?",
    parquet_file=f"{wd}/stations",
    limit=1000,
    update=True)
variables = meteocat_utils.get_complete_socrata_dataset(
    url="https://analisi.transparenciacatalunya.cat/resource/4fb2-n3yi.csv?",
    parquet_file=f"{wd}/variables",
    limit=1000,
    update=True)
timeseries = meteocat_utils.get_complete_socrata_dataset(
    url="https://analisi.transparenciacatalunya.cat/resource/nzvn-apee.csv?",
    parquet_file=f"{wd}/timeseries",
    limit=500000,
    update=True,
    one_file=False)
