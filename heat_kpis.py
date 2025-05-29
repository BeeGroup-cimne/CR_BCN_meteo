import os
import polars as pl
import numpy as np
from pythermalcomfort.models import heat_index_lu
from scipy.interpolate import RegularGridInterpolator

# Filepath for caching
parquet_file = "heat_index_lookup.parquet"

def generate_heat_index_table():
    # Define ranges
    tdb_range = np.arange(-20, 61)  # °C
    rh_range = np.arange(0, 101)    # %

    # Compute heat index matrix
    HI = np.full((len(tdb_range), len(rh_range)), np.nan)
    for i, t in enumerate(tdb_range):
        for j, rh in enumerate(rh_range):
            HI[i, j] = heat_index_lu(tdb=t, rh=rh).hi

    # Store in long-form Polars DataFrame
    df = pl.DataFrame({
        "tdb": np.repeat(tdb_range, len(rh_range)),
        "rh": np.tile(rh_range, len(tdb_range)),
        "heat_index": HI.flatten()
    })
    df.write_parquet(parquet_file)
    return df, tdb_range, rh_range, HI

def load_or_generate_lookup():
    if os.path.exists(parquet_file):
        df = pl.read_parquet(parquet_file)
        tdb_range = np.unique(df["tdb"])
        rh_range = np.unique(df["rh"])
        HI = df.to_pandas().pivot(index="tdb", columns="rh", values="heat_index").values
    else:
        df, tdb_range, rh_range, HI = generate_heat_index_table()
    return df, tdb_range, rh_range, HI

# Load or generate
df_lookup, tdb_vals, rh_vals, hi_grid = load_or_generate_lookup()

# Interpolator function
interpolator = RegularGridInterpolator((tdb_vals, rh_vals), hi_grid, bounds_error=False, fill_value=None)

def interpolate_heat_index(tdb_list, rh_list):
    """Vectorized interpolation of heat index from precomputed table"""
    points = np.column_stack((tdb_list, rh_list))
    return interpolator(points)