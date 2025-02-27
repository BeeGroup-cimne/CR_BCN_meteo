from utils import *

##########
# Settings
##########

data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)
update = False
ym_range = [200801, 201712]
lat_range = [42.9, 40.5]
lon_range = [0.1, 3.4]

# ERA5Land
weather = load_ERA5Land(
    data_dir = f"{data_dir}/era5land",
    lat_range = lat_range,
    lon_range = lon_range,
    ym_range = ym_range,
    update = update)

import polars as pl

