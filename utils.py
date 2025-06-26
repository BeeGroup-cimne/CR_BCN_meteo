from catboost import CatBoostRegressor, Pool
import requests
import os
import sys
import xarray as xr
from functools import reduce
import pvlib
import pandas as pd
from shapely.geometry import Point, box
import datetime
from math import pi
from sklearn.model_selection import train_test_split
import zipfile
import cdsapi
import cfgrib
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.tri import Triangulation
import rasterio
from rasterio.mask import mask
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import box
from shapely import union_all
from rasterstats import zonal_stats
from matplotlib.backends.backend_pdf import PdfPages
from hypercadaster_ES import mergers
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objs as go
from plotly.offline import plot

def get_gdf_buffed_area(barcelona_shapefile_directory, catalonia_shapefile_directory, distance):
    # Distance in meters /!\
    gdf_b = gpd.read_file(barcelona_shapefile_directory)
    gdf_c = gpd.read_file(catalonia_shapefile_directory)
    crs_b = gdf_b.crs
    # Make sure the coordinate reference system is the same
    if crs_b != gdf_c.crs:
        gdf_c = gdf_c.to_crs(crs_b)
    # Union on multipolygons
    gdf_b = gdf_b.geometry[
        gdf_b.geometry.notnull() &
        gdf_b.geometry.is_valid &
        ~gdf_b.geometry.is_empty &
        gdf_b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
    gdf_c = gdf_c.geometry[
        gdf_c.geometry.notnull() &
        gdf_c.geometry.is_valid &
        ~gdf_c.geometry.is_empty &
        gdf_c.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
    # Apply union_all safely
    barcelona_polygon = union_all(gdf_b.geometry)
    catalonia_polygon = union_all(gdf_c.geometry)
    # Convert polygon to GeoDataFrame (first using GeoSeries)
    barcelona_geoseries = gpd.GeoSeries([barcelona_polygon], crs=crs_b)
    barcelona_gdf = gpd.GeoDataFrame(geometry=barcelona_geoseries)
    catalonia_geoseries = gpd.GeoSeries([catalonia_polygon], crs=crs_b)
    catalonia_gdf = gpd.GeoDataFrame(geometry=catalonia_geoseries)
    # Converting to UTM to correctly buffer 10km
    barcelona_utm = barcelona_gdf.to_crs(epsg=32631)
    # Buffer
    barcelona_buffered = barcelona_utm.buffer(distance)
    # Converting back crs and object type
    barcelona_buffered = barcelona_buffered.to_crs(crs_b)
    barcelona_gdf = gpd.GeoDataFrame(geometry=barcelona_buffered)

    final = gpd.overlay(barcelona_gdf, catalonia_gdf, how='intersection')

    return final


def filter_weather_station(ds, polygon, crs_type):
    weather_stations = ds['weatherStation'].values
    coordinates = []
    for station in weather_stations:
        lat, lon = map(float, station.split("_"))
        coordinates.append((lat, lon))
    stations_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in coordinates], crs=crs_type)
    indices_in_area = stations_gdf[stations_gdf.within(polygon)].index
    filtered_data = ds['weatherStation'][indices_in_area].values
    new_ds = ds.sel(weatherStation=filtered_data)
    return new_ds


def unzip_zarr_file(file_path):
    if not os.path.exists(file_path):
        if file_path[-4:] == 'zarr':
            with zipfile.ZipFile(file_path[:-4] + 'zip', 'r') as zip_ref:
                zip_ref.extractall(file_path)


def get_dataset(high_res_zarr_dir, low_res_hist_zarr_dir, low_res_fore_zarr_dir, static_features_zarr_file,
                barcelona_shp_dir, catalonia_shp_dir, hypercadaster_ES_dir,
                low_res_bbox_polygon=None, plots=False, k_fore=0, plots_dir="plots"):
    ## Unzipping data if not unzipped
    for file in [high_res_zarr_dir, low_res_hist_zarr_dir, low_res_fore_zarr_dir, static_features_zarr_file]:
        if file.endswith(".zip"):
            unzip_zarr_file(file)

    ## Open data
    high_res_ds = xr.open_zarr(high_res_zarr_dir, chunks={})
    low_res_hist_ds = xr.open_zarr(low_res_hist_zarr_dir, chunks={})
    low_res_fore_ds = xr.open_zarr(low_res_fore_zarr_dir, chunks={})

    ## Filter out the unnecessary weather stations (out of scope)
    if low_res_bbox_polygon is not None:
        low_res_fore_ds = filter_weather_station(low_res_fore_ds, low_res_bbox_polygon, "EPSG:4326")
        low_res_hist_ds = filter_weather_station(low_res_hist_ds, low_res_bbox_polygon, "EPSG:4326")
    else:
        barcelona_area = get_gdf_buffed_area(barcelona_shp_dir, catalonia_shp_dir, 5000)
        low_res_bbox_polygon = barcelona_area.values.union_all()
        low_res_fore_ds = filter_weather_station(low_res_fore_ds, low_res_bbox_polygon, "EPSG:4326")
        low_res_hist_ds = filter_weather_station(low_res_hist_ds, low_res_bbox_polygon, "EPSG:4326")
    barcelona_area = gpd.read_file(barcelona_shp_dir)
    barcelona_regular_polygon = barcelona_area.union_all()
    high_res_ds = filter_weather_station(high_res_ds, barcelona_regular_polygon, "EPSG:4326")

    weather_station_hr_ids = high_res_ds.weatherStation.values.tolist()
    weather_coords_hr = [tuple(map(float, ws.split('_'))) for ws in weather_station_hr_ids]  # (lat, lon)
    weather_points_hr = [Point(lon, lat) for lat, lon in weather_coords_hr]
    if os.path.exists(static_features_zarr_file):
        static_features_ds = xr.open_zarr(static_features_zarr_file, chunks={})
    else:
        static_features_df = compute_static_features_maps(
            barcelona_geom = barcelona_regular_polygon,
            centroids=weather_points_hr,
            hypercadaster_ES_dir=hypercadaster_ES_dir,
            buffer=100,
            shape="square",
            utm_crs="EPSG:25831"
        )
        static_features_ds = static_features_df.drop(
            columns=["building_area_urbanization_and_landscaping_works_undeveloped_land_percentile","geometry"],
            errors='ignore').to_xarray()
        static_features_ds.to_zarr(static_features_zarr_file, mode="w")
    if plots:
        # Plot static features
        plot_static_features_maps(
            static_features_ds,
            markersize=2,
            output_pdf=f"{plots_dir}/static_features_map.pdf"
        )
        plot_weather_stations_and_bbox(
            low_res_hist_ds if k_fore==0 else low_res_fore_ds,
            high_res_ds,
            barcelona_area,
            low_res_bbox_polygon,
            output_pdf=f"{plots_dir}/weather_map.pdf"
        )

    return high_res_ds, low_res_fore_ds, low_res_hist_ds, static_features_ds

def plot_weather_stations_and_bbox(
    low_res_hist_ds,
    high_res_ds,
    barcelona_area,
    low_res_bbox_polygon,
    output_pdf
):
    """
    Plot Barcelona area, bounding box, and weather stations (low- and high-resolution),
    saving the result to a PDF.

    Parameters:
        low_res_hist_ds: xarray Dataset containing low-resolution weather station info.
        high_res_ds: xarray Dataset containing high-resolution weather station info.
        barcelona_area: GeoDataFrame of the Barcelona boundary.
        low_res_bbox_polygon: Shapely Polygon representing the bounding box.
        output_pdf: Path to the output PDF file.
    """

    # Step 1: Parse low-res weather station coordinates
    weather_station_ids = low_res_hist_ds.weatherStation.values.tolist()
    weather_coords = [tuple(map(float, ws.split('_'))) for ws in weather_station_ids]  # (lat, lon)
    weather_points = [Point(lon, lat) for lat, lon in weather_coords]
    weather_gdf = gpd.GeoDataFrame(geometry=weather_points, crs="EPSG:4326")

    # Step 2: Reproject low-res weather stations to match CRS
    if barcelona_area.crs != weather_gdf.crs:
        weather_gdf = weather_gdf.to_crs(barcelona_area.crs)

    # Step 3: Parse high-res weather station coordinates
    weather_station_hr_ids = high_res_ds.weatherStation.values.tolist()
    weather_coords_hr = [tuple(map(float, ws.split('_'))) for ws in weather_station_hr_ids]
    weather_points_hr = [Point(lon, lat) for lat, lon in weather_coords_hr]
    weather_gdf_hr = gpd.GeoDataFrame(geometry=weather_points_hr, crs="EPSG:4326")

    # Step 4: Reproject high-res weather stations to match CRS
    if barcelona_area.crs != weather_gdf_hr.crs:
        weather_gdf_hr = weather_gdf_hr.to_crs(barcelona_area.crs)

    # Step 5: Convert bbox polygon to GeoDataFrame
    low_res_gdf = gpd.GeoDataFrame(geometry=[low_res_bbox_polygon], crs=barcelona_area.crs)

    # Step 6: Plot and save to PDF
    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(20, 20))

        barcelona_area.plot(ax=ax, color='white', edgecolor='black', label='Barcelona Area')
        low_res_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2, label='Low-Res BBox')
        weather_gdf_hr.plot(ax=ax, color='lightblue', marker='o', alpha=0.6, markersize=2, label='Weather Stations HR')
        weather_gdf.plot(ax=ax, color='green', marker='o', markersize=80, label='Weather Stations LR')

        ax.legend(fontsize=20)
        ax.set_title("Barcelona Area, Bounding Box, and Weather Stations", fontsize=26)
        ax.tick_params(labelsize=20)

        pdf.savefig(fig)
        plt.close(fig)

def compute_static_features_maps(barcelona_geom, centroids, hypercadaster_ES_dir, buffer=100, shape="square", utm_crs="EPSG:25831"):
    """
    Create a GeoDataFrame of polygons (square or circular) centered on Point geometries.

    Parameters:
        centroids (list of shapely.geometry.Point): List of points (lon, lat).
        buffer (float): Buffer distance (in meters).
        shape (str): 'square' or 'circle' buffer shape.
        utm_crs (str): Projected CRS to use for buffering (e.g., EPSG:25831 for Barcelona).

    Returns:
        GeoDataFrame with buffered polygons and index column.
    """
    # Create GeoDataFrame of points in WGS84
    gdf_points = gpd.GeoDataFrame(geometry=centroids, crs="EPSG:4326")

    # Add lat/lon columns (from Point geometries)
    gdf_points["lat"] = gdf_points.geometry.y.round(6)
    gdf_points["lon"] = gdf_points.geometry.x.round(6)
    gdf_points["index"] = gdf_points["lat"].astype(str) + "_" + gdf_points["lon"].astype(str)

    # Reproject to UTM for accurate buffering
    gdf_points = gdf_points.to_crs(utm_crs)

    # Create buffered polygons
    if shape == "circle":
        gdf = gdf_points.copy()
        gdf["geometry"] = gdf.geometry.buffer(buffer)
    elif shape == "square":
        half = buffer / 2
        gdf = gdf_points.copy()
        gdf["geometry"] = gdf.geometry.apply(
            lambda p: box(p.x - half, p.y - half, p.x + half, p.y + half)
        )
    else:
        raise ValueError("shape must be either 'square' or 'circle'")

    gdf.drop(columns=["lat", "lon"], inplace=True)

    # Read the Digital Elevation Model
    gdf["centroid"] = gdf.centroid
    gdf.set_geometry("centroid", inplace=True)
    gdf = mergers.join_DEM_raster(gdf, f"{hypercadaster_ES_dir}/DEM_rasters")
    gdf["elevation"] = compute_percentile_ranks(gdf[["elevation"]])["elevation_percentile"]

    # Read the cadaster info
    gdf.set_geometry("geometry", inplace=True)
    gdf_cadaster = pd.read_pickle(f"{hypercadaster_ES_dir}/08900_only_addresses.pkl", compression="gzip")
    gdf_cadaster = gdf_cadaster.drop_duplicates(subset=["building_reference"])
    gdf_cadaster.set_geometry("building_centroid", inplace=True)
    gdf_cadaster = gdf_cadaster.to_crs(gdf.crs)
    joined = gpd.sjoin(gdf_cadaster, gdf[['geometry']], how='left', predicate='within')
    agg_dict = {
        'building_area_commercial': 'sum',
        'building_area_residential': 'sum',
        'building_area_warehouse_parking': 'sum',
        'building_area_offices': 'sum',
        'building_area_singular_building': 'sum',
        'building_area_cultural': 'sum',
        'building_area_entertainment_venues': 'sum',
        'building_area_industrial': 'sum',
        'building_area_urbanization_and_landscaping_works_undeveloped_land': 'sum',
        'building_area_leisure_and_hospitality': 'sum',
        'building_area_sports_facilities': 'sum',
        'building_area_religious': 'sum',
        'building_area_healthcare_and_charity': 'sum',
        'n_dwellings': 'sum',
        'n_floors_above_ground': 'mean'
    }
    aggregated = joined.groupby('index_right').agg(agg_dict)
    aggregated = aggregated.reindex(gdf.index).fillna(0)
    aggregated.fillna(0.0, inplace=True)
    aggregated = compute_percentile_ranks(aggregated)
    aggregated = aggregated[[col for col in aggregated.columns if col.endswith('_percentile')]]
    gdf = gdf.join(aggregated, how='left')

    # Read the NDVI
    raster_NDVI = read_ndvi_from_gpkg_scaled(
        f"{hypercadaster_ES_dir}/NDVI/NDVI.gpkg",
        transform_bbox(barcelona_geom.bounds, "EPSG:4326", "EPSG:25831"))
    #plot_ndvi(raster_NDVI[0], title="NDVI Escalado", save_path="ndvi.png")
    gdf = extract_ndvi_stats_from_array(gdf, raster_NDVI, 0)

    gdf.set_index("index", inplace=True)
    gdf.drop(columns=["geometry", "centroid"], inplace=True)
    gdf = normalize_0_1(gdf)

    return gdf

def normalize_0_1(df):
    """
    Normalizes all numeric columns in a DataFrame to the [0, 1] range using min-max scaling.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Copy of the DataFrame with normalized numeric columns.
    """
    df_norm = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val == max_val:
            # Avoid division by zero; assign 0 if constant column
            df_norm[col] = 0.0
        else:
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)

    return df_norm

def compute_percentile_ranks(df):
    """
    Computes ECDF-style percentile ranks for all numeric columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame (or GeoDataFrame).

    Returns:
        pd.DataFrame: Copy with new columns named <original_column>_percentile (values in [0, 1]).
    """
    percentile_df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        values = df[col].values
        n = len(values)

        if n <= 1 or np.all(values == values[0]):
            # Constant or single value column
            percentile_df[col + "_percentile"] = 0.0
        else:
            # Compute ECDF-style percentile rank: proportion of values less than or equal to each
            # using strict "<" to ensure values are in [0, 1)
            ranks = np.sum(values[:, None] >= values[None, :], axis=1) / (n - 1)
            percentile_df[col + "_percentile"] = ranks

    return percentile_df

def extract_ndvi_stats_from_array(gdf, raster_NDVI, nodata_val=-1.0):
    """
    Compute min, max, and mean NDVI from a masked array over polygons in a GeoDataFrame.

    Parameters:
    - gdf (GeoDataFrame): A GeoDataFrame with polygon geometries.
    - raster_NDVI (tuple): A tuple of (masked_array, affine_transform).
    - nodata_val (float): Value in the array that represents nodata (default: -1.0).

    Returns:
    - GeoDataFrame: Same as input `gdf` with ndvi_min, ndvi_max, and ndvi_mean columns added.
    """
    data, transform = raster_NDVI

    # Replace nodata value with np.nan
    array = np.where(data == nodata_val, np.nan, data)

    # Compute zonal statistics
    stats = zonal_stats(
        vectors=gdf,
        raster=array,
        affine=transform,
        stats=["min", "max", "mean"],
        nodata=np.nan
    )

    # Add stats to the GeoDataFrame
    gdf = gdf.copy()
    gdf["ndvi_min"] = [s["min"] for s in stats]
    gdf["ndvi_max"] = [s["max"] for s in stats]
    gdf["ndvi_mean"] = [s["mean"] for s in stats]

    return gdf

def plot_ndvi(ndvi_array, title="NDVI", cmap="RdYlGn", save_path=None):
    ndvi_plot = np.asarray(ndvi_array)
    if ndvi_plot.ndim > 2:
        ndvi_plot = ndvi_plot.squeeze()

    plt.figure(figsize=(10, 8))
    img = plt.imshow(ndvi_plot, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(img, label="NDVI value")
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot guardado en {save_path}")

    plt.show()

def transform_bbox(bbox, src_crs, dst_crs):
    """
    Transform bounding box coordinates from src_crs to dst_crs.

    Parameters:
    - bbox: tuple (xmin, ymin, xmax, ymax)
    - src_crs: source CRS as EPSG code string, e.g., "EPSG:4326"
    - dst_crs: destination CRS as EPSG code string, e.g., "EPSG:25831"

    Returns:
    - transformed_bbox: tuple (xmin, ymin, xmax, ymax) in destination CRS
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    xmin, ymin = transformer.transform(bbox[0], bbox[1])
    xmax, ymax = transformer.transform(bbox[2], bbox[3])

    # Make sure the bounds are ordered correctly after transformation
    xmin_new, xmax_new = sorted([xmin, xmax])
    ymin_new, ymax_new = sorted([ymin, ymax])

    return (xmin_new, ymin_new, xmax_new, ymax_new)

def read_ndvi_from_gpkg_scaled(gpkg_path, bbox):
    """
    Lee un raster NDVI de un archivo GPKG, recorta por un bounding box,
    y escala los valores de [0,200] a [-1,1].
    """
    bbox_polygon = box(*bbox)
    geojson_geom = [bbox_polygon.__geo_interface__]

    with rasterio.open(gpkg_path) as src:
        out_image, out_transform = mask(src, geojson_geom, crop=True)
        ndvi_raw = out_image[0]  # suponer que es una sola banda

        # Enmascarar valores NoData
        ndvi_masked = np.ma.masked_equal(ndvi_raw, src.nodata)

        # Escalar de [0, 200] a [-1, 1]
        ndvi_scaled = (ndvi_masked / 200.0) * 2 - 1

        # Enmascarar valores fuera del rango [-1, 1] si es necesario
        ndvi_scaled = np.ma.masked_outside(ndvi_scaled, -1, 1)

    return ndvi_scaled, out_transform


def minmax_normalization(ds):
    max_vars = {}
    min_vars = {}
    ds = ds.chunk({"time": -1})
    for var in ds.data_vars:
        if var != 'x' and var != 'y':
            print(f'Normalizing {var}')
            # Compute 5th and 95th quantiles along the "time" dimension
            q_min = ds[var].quantile(0.05, dim="time")
            q_max = ds[var].quantile(0.95, dim="time")

            # Store the min/max for inverse transformation if needed
            min_vars[var] = q_min
            max_vars[var] = q_max
            min_vars[var] = min_vars[var].drop_vars('quantile')
            max_vars[var] = max_vars[var].drop_vars('quantile')
            ds[var] = (ds[var] - q_min) / (q_max - q_min)  # On part du principe que max et min sont différents
            ds[var] = ds[var].drop_vars('quantile')
    ds = ds.drop_vars('quantile')
    return ds, min_vars, max_vars


def format_input_ds(input_ds, static_features_ds):
    # Formatting input : merging static high-res and dynamic low-res data
    weather_stations = input_ds['weatherStation'].values
    new_vars = {}
    for var in input_ds.data_vars:
        for station in weather_stations:
            new_var_name = f"{var}_{station}"
            new_vars[new_var_name] = input_ds[var].sel(weatherStation=station).drop_vars('weatherStation')
    input_ds = xr.Dataset(new_vars)
    del new_vars, new_var_name, station, var, weather_stations
    static_features_ds = static_features_ds.rename({'index': 'weatherStation'})
    input_expanded = input_ds.expand_dims(
        {"weatherStation": len(static_features_ds.coords["weatherStation"])})
    input_expanded["weatherStation"] = static_features_ds.coords["weatherStation"]
    static_features_expanded = static_features_ds.expand_dims({"time": len(input_ds.coords["time"])})
    static_features_expanded["time"] = input_ds.coords["time"]
    del input_ds, static_features_ds
    input_ds = xr.merge([input_expanded, static_features_expanded])
    del input_expanded, static_features_expanded
    input_ds = input_ds.unify_chunks()
    return input_ds


def plot_static_features_maps(dataset, crs="EPSG:4326", cmap="viridis", markersize=100, output_pdf="static_feature_maps.pdf"):
    """
    Plots color-coded maps for each variable in a static features xarray.Dataset and saves them in a single PDF.

    Parameters:
        dataset (xarray.Dataset or pd.DataFrame): Dataset with index in 'lat_lon' format.
        crs (str): Coordinate Reference System (default: EPSG:4326).
        cmap (str): Colormap for plotting (default: 'viridis').
        markersize (int): Marker size for plot points.
        output_pdf (str): Output PDF file name (default: 'static_feature_maps.pdf').
    """
    # Convert xarray.Dataset to pandas.DataFrame if needed
    if isinstance(dataset, xr.Dataset):
        df = dataset.to_dataframe()
    elif isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
    else:
        raise TypeError("Input must be an xarray.Dataset or pandas.DataFrame")

    # Ensure index is string-formatted like 'lat_lon'
    try:
        df['lat'] = df.index.map(lambda x: float(str(x).split('_')[0]))
        df['lon'] = df.index.map(lambda x: float(str(x).split('_')[1]))
    except Exception as e:
        raise ValueError("Index format must be 'lat_lon', e.g., '41.38_2.17'") from e

    # Build GeoDataFrame
    geometry = [Point(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
    gdf = gpd.GeoDataFrame(df.drop(columns=['lat', 'lon']), geometry=geometry, crs=crs)

    # Save all plots to a single PDF
    with PdfPages(output_pdf) as pdf:
        for column in gdf.columns.difference(['geometry']):
            fig, ax = plt.subplots(figsize=(8, 6))
            gdf.plot(
                ax=ax,
                column=column,
                cmap=cmap,
                legend=True,
                markersize=markersize
            )
            ax.set_title(f"{column.replace('_', ' ').title()}", fontsize=14)
            ax.set_axis_off()
            pdf.savefig(fig)  # Save the current figure into the PDF
            plt.close(fig)

    print(f"All plots saved to '{output_pdf}'")

def transformation_general(high_res_ds, low_res_ds, static_features_ds, time_indices, fore):
    # Dropping redundant variables in inputs
    coords = high_res_ds[["x", "y"]].isel(time=0, weatherStation=slice(0, 10222))
    coords = coords.to_dataframe().reset_index()
    coords = coords.drop_duplicates()
    coords = coords.to_numpy()
    high_res_ds = high_res_ds.drop_vars(["x", "y"])
    dropped_vars = ["windSpeedEast", "windSpeedNorth"] if fore else ["dewAirTemperature", "windSpeedEast", "windSpeedNorth"]
    low_res_ds = low_res_ds.drop_vars(dropped_vars)

    # Data normalization: minmax normalization
    low_res_ds, low_res_min, low_res_max = minmax_normalization(low_res_ds)
    high_res_ds, high_res_min, high_res_max = minmax_normalization(high_res_ds)
    # static feature normalization made in pre-treatment of static data

    # keeping time indices from chosen chunk (randomized)
    high_res_ds, low_res_ds = [ds.isel(time=time_indices) for ds in [high_res_ds, low_res_ds]]

    low_res_ds = format_input_ds(low_res_ds, static_features_ds)

    return high_res_ds, low_res_ds, high_res_min, high_res_max, low_res_min, low_res_max, coords


def data_split(X_input, Y_output, test_ratio, nspace):
    ntime = int(X_input.shape[0] / nspace)
    if X_input.shape[0] != nspace * ntime:
        print("ERROR : INCONSISTENT SHAPE!")
        return
    if X_input.shape[0] != Y_output.shape[0]:
        print("BOTH DATAFRAME SHOULD HAVE SAME SHAPE!")
        return

    unique_times = X_input.index.get_level_values('time').unique()
    train_times, test_times = train_test_split(unique_times, test_size=test_ratio, random_state=42)

    def sort_df(ttimes, XY):
        # Step 1: Convert train_times to a DataFrame with a sorting key
        ttimes = pd.DataFrame({'time': ttimes, 'sort_order': range(len(ttimes))})
        # Step 2: Reset X_train's index to access 'time' for merging
        XY.reset_index(inplace=True)
        # Step 3: Merge X_train with train_times_df to assign sorting order
        XY = XY.merge(ttimes, on='time', how='left', copy=False)
        # Step 4: Sort the DataFrame by 'sort_order'
        XY.sort_values(by='sort_order', inplace=True)
        XY = XY.drop(columns=['sort_order'])
        # Step 5: Set the original multiindex back
        XY.set_index(['time', 'weatherStation'], inplace=True)
        return XY
    # creating new dataframes
    # X_test getting right time
    X_test = X_input.loc[X_input.index.get_level_values('time').isin(test_times)]
    # X_test sorting
    X_test = sort_df(test_times, X_test)
    # X_train getting right time
    X_input = X_input.loc[X_input.index.get_level_values('time').isin(train_times)]
    # X_train sorting
    X_input = sort_df(train_times, X_input)
    # Same for Y_test and Y_train
    Y_test = Y_output.loc[Y_output.index.get_level_values('time').isin(test_times)]
    Y_test = sort_df(test_times, Y_test)
    Y_output = Y_output.loc[Y_output.index.get_level_values('time').isin(train_times)]
    Y_output = sort_df(train_times, Y_output)

    return X_input, X_test, Y_output, Y_test, train_times, test_times


def save_info(ds, high_min, high_max, low_min, low_max, nextcloud_local_dir):
    xylatlon = ds[["x", "y"]].isel(time=0, weatherStation=slice(0, 10222))
    xylatlon = xylatlon.to_dataframe().reset_index()
    xylatlon = xylatlon.drop(columns=["time"])
    xylatlon = xylatlon.drop_duplicates()
    xylatlon = xylatlon.to_numpy()
    np.save(f'{nextcloud_local_dir}/xylatlon.npy', xylatlon)
    names = ['high_min', 'high_max', 'low_min', 'low_max']
    for i, ar in enumerate([high_min, high_max, low_min, low_max]):
        ar = np.stack([data.values for data in ar.values()], axis=1)
        np.save(f'{nextcloud_local_dir}/{names[i]}.npy', ar)


def minmax_norm_realtime(ds, low_res_min, low_res_max):
    normalized_data_vars = {}
    for ivar, var in enumerate(ds.data_vars):
        max = low_res_max[:, ivar]
        min = low_res_min[:, ivar]
        normalized_data_vars[var] = (ds[var] - min) / (max - min)  # On part du principe que max et min sont différents
    normalized_ds = xr.Dataset(normalized_data_vars, coords=ds.coords)
    return normalized_ds


def MeteoGalicia_variables():
    return [
        {
            'RawName': 'u',
            'name': 'windSpeedEast'
        },
        {
            'RawName': 'lat',
            'name': 'latitude'
        },
        {
            'RawName': 'lon',
            'name': 'longitude'
        },
        {
            'RawName': 'v',
            'name': 'windSpeedNorth'
        },
        {
            'RawName': 'temp',
            'name': 'airTemperature'
        },
        {
            'RawName': 'prec',
            'name': 'totalPrecipitation'
        },
        {
            'RawName': 'swflx',
            'name': 'GHI'
        },
        {
            'RawName': 'rh',
            'name': 'relativeHumidity'
        }
        ]


def MeteoGalicia_transformation(df):
    #                       Cleaning and renaming
    if 'index' in df.columns:
        df = df.drop(['index'])
    df = df.drop_nulls()

    renamed_columns = {x['RawName'] : x['name'] for x in MeteoGalicia_variables()}
    renamed_columns_filtered = {k: v for k, v in renamed_columns.items() if k in df.columns}
    df = df.rename(renamed_columns_filtered)

    #                       Formatting
    df = df.sort(["latitude", "longitude", "time"])

    # Transform time
    df = df.with_columns(pl.col("time").cast(pl.Datetime("ns", None)))

    # totalPrecipitation in mm same as in kg.m-2 : it isn't a daily accumulated data in the MeteoGalicia datasets
    # no need to make it hourly accumulated, it already is
    # GHI is already hourly accumulated and in Wm-2

    # Conversions and calculations : windSpeed, windDirection, airTemperature
    df = df.with_columns([
        np.sqrt(pl.col("windSpeedEast") ** 2 + pl.col("windSpeedNorth") ** 2)
        .alias("windSpeed"),
        ((180 + np.degrees(np.arctan2(pl.col("windSpeedEast"), pl.col("windSpeedNorth")))) % 360)
        .alias("windDirection"),
        (pl.col("airTemperature") - 273.15).alias("airTemperature")
    ])

    # Value replaced by average with next value (is the shift in time visible?-->no, assigned to previous timestep)
    # On which variables is this shift necessary (I thought on all - ask G.M.) (maybe except the accumulated ones)
    for var in ["windSpeed", "windSpeedEast", "windSpeedNorth", "windDirection",
                "airTemperature", "relativeHumidity"]:
        df = df.with_columns(
            pl.when(pl.col("time") < df.select(pl.col("time").max()).item())
            .then((pl.col(var).shift(-1) + pl.col(var)) / 2)
            .otherwise(pl.col(var))
            .alias(f"{var}_avg")
        )

    # Add the weatherStation column
    df = df.with_columns(
        pl.format("{}_{}", pl.col("latitude").round(6), pl.col("longitude").round(6)).alias("weatherStation")
    )

    # Drop the columns that have been averaged
    df = df.drop(["windSpeed", "windSpeedEast", "windSpeedNorth", "windDirection",
                 "airTemperature", "relativeHumidity"])
    rename_map = {col: col.replace("_avg", "") for col in df.columns if col.endswith("_avg")}
    df = df.rename(rename_map)

    def join_solar_data(df):

        dfp = df.to_pandas()
        location = pvlib.location.Location(
            latitude=df.select("latitude").unique().item(),
            longitude=df.select("longitude").unique().item())
        solar_df = location.get_solarposition(
            dfp['time'] + pd.Timedelta(minutes=30),
            temperature=dfp['airTemperature']).reset_index()
        dni = pvlib.irradiance.disc(
            ghi=dfp["GHI"],
            solar_zenith=solar_df['apparent_zenith'],
            datetime_or_doy=solar_df['time'].dt.dayofyear)
        rad_df = pvlib.irradiance.complete_irradiance(
            solar_zenith=solar_df['apparent_zenith'],
            ghi=dfp["GHI"],
            dni=dni["dni"],
            dhi=None).rename(columns={'ghi':'GHI','dni':'DNI','dhi':'DHI'})
        solar_df = solar_df.drop(['apparent_zenith', 'zenith', 'apparent_elevation', 'equation_of_time'], axis=1)
        solar_df = solar_df.rename(columns={'elevation': 'sunElevation', 'azimuth': 'sunAzimuth'})
        solar_df['time'] = solar_df['time'] - pd.Timedelta(minutes=30)
        solar_df = pl.from_pandas(pd.concat([solar_df, rad_df], axis=1))

        return solar_df.join(df.drop("GHI"), on="time", how="inner")

    # Loop through each group and apply the join_solar_data function
    result_list = []
    for group_name, group_df in df.group_by("weatherStation"):
        result = join_solar_data(group_df)
        result_list.append(result)
    df = pl.concat(result_list)

    # Dropping latitude and longitude, already found in weatherStation column
    df = df.drop(["latitude", "longitude"])

    return df


def ERA5Land_variables():
    return [
        {
            'RawName': 'u10',
            'name': 'windSpeedEast'
        },
        {
            'RawName': 'v10',
            'name': 'windSpeedNorth'
        },
        {
            'RawName': 't2m',
            'name': 'airTemperature'
        },
        {
            'RawName': 'd2m',
            'name': 'dewAirTemperature'
        },
        {
            'RawName': 'lai_lv',
            'name': 'lowVegetationRatio'
        },
        {
            'RawName': 'lai_hv',
            'name': 'highVegetationRatio'
        },
        {
            'RawName': 'tp',
            'name': 'totalPrecipitation'
        },
        {
            'RawName': 'ssrd',
            'name': 'GHI'
        },
        {
            'RawName': 'fal',
            'name': 'albedo'
        },
        {
            'RawName': 'stl4',
            'name': 'soilTemperature'
        },
        {
            'RawName': 'swvl4',
            'name': 'soilWaterRatio'
        }
        ]


def ERA5Land_transformation(df):
    #                       Cleaning and renaming
    df = df.drop_nulls()

    df = df.drop(['time', 'step', 'number', 'surface', 'depthBelowLandLayer', 'time_right','step_right','number_right'], strict=False)

    renamed_columns = {**{'valid_time': 'time'},
                       **{x['RawName']: x['name'] for x in ERA5Land_variables()}}
    renamed_columns_filtered = {k: v for k, v in renamed_columns.items() if k in df.columns}
    df = df.rename(renamed_columns_filtered)

    #                       Formatting
    df = df.sort(["latitude", "longitude", "time"])

    # Transform time
    # df = df.with_columns(pl.col("time").dt.replace_time_zone("UTC"))  # Localize to UTC
    df = df.with_columns(pl.col("time").cast(pl.Datetime("ns", None)))

    # Transform GHI and total precipitation (from accumulated values to instant values)
    df = df.with_columns(
        pl.when(pl.col("time") == df.select(pl.col("time").min()).item())
        .then(0)
        .when(pl.col("time") == df.select(pl.col("time").max()).item())
        .then(0)
        .otherwise((pl.col("GHI").shift(-1) - pl.col("GHI")) / 3600)
        .alias("GHI_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("time") == df.select(pl.col("time").min()).item())
        .then(0)
        .when(pl.col("time") == df.select(pl.col("time").max()).item())
        .then(pl.col("totalPrecipitation") - pl.col("totalPrecipitation").shift())
        .otherwise((pl.col("totalPrecipitation").shift(-1) - pl.col("totalPrecipitation")) * 1000)
        .alias("totalPrecipitation_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("GHI_avg") > 0)
        .then(pl.col("GHI_avg"))
        .otherwise(0)
        .alias("GHI_avg")
    )
    df = df.with_columns(
        pl.when(pl.col("totalPrecipitation_avg") >= 0)
        .then(pl.col("totalPrecipitation_avg"))
        .otherwise(pl.col("totalPrecipitation_avg").shift())
        .alias("totalPrecipitation_avg")
    )
    # Transform temperature, wind variables and humidity
    df = df.with_columns([
        np.sqrt(pl.col("windSpeedEast") ** 2 + pl.col("windSpeedNorth") ** 2)
        .alias("windSpeed"),
        ((180 + np.degrees(np.arctan2(pl.col("windSpeedEast"), pl.col("windSpeedNorth")))) % 360)
        .alias("windDirection"),
        (pl.col("soilTemperature") - 273.15).alias("soilTemperature"),
        (pl.col("dewAirTemperature") - 273.15).alias("dewAirTemperature"),
        (pl.col("airTemperature") - 273.15).alias("airTemperature")
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = df.with_columns(
            pl.when(pl.col("airTemperature").is_null() | pl.col("dewAirTemperature").is_null())
            .then(None)
            .otherwise(
                (np.exp(17.67 * pl.col("dewAirTemperature") / (pl.col("dewAirTemperature") + 243.5)) /
                np.exp(17.67 * pl.col("airTemperature") / (pl.col("airTemperature") + 243.5)))
            ).alias("relativeHumidity")
        )

    # shifting the timestamps to XX:30 (instead of round hours XX:00) : is it really shifting the timestamp and not just
    # changing the value to an avg ?
    for var in ["windSpeed", "windDirection", "soilTemperature", "dewAirTemperature", "airTemperature",
                "relativeHumidity", "windSpeedEast", "windSpeedNorth",
                "lowVegetationRatio", "highVegetationRatio", "albedo", "soilWaterRatio"]:
        df = df.with_columns(
            pl.when(pl.col("time") < df.select(pl.col("time").max()).item())
            .then((pl.col(var).shift(-1) + pl.col(var)) / 2)
            .otherwise(pl.col(var))
            .alias(f"{var}_avg")
        )

    # Add the weatherStation column
    df = df.with_columns(
        pl.format("{}_{}", pl.col("latitude").round(6), pl.col("longitude").round(6)).alias("weatherStation")
    )

    # Drop some columns
    df = df.drop(["windSpeed", "windDirection", "soilTemperature", "dewAirTemperature", "airTemperature",
                "relativeHumidity", "GHI", "totalPrecipitation", "windSpeedEast", "windSpeedNorth",
                "lowVegetationRatio", "highVegetationRatio", "albedo", "soilWaterRatio"])
    rename_map = {col: col.replace("_avg", "") for col in df.columns if col.endswith("_avg")}
    df = df.rename(rename_map)

    # Add solar position and solar radiation components
    def join_solar_data(df):

        dfp = df.to_pandas()
        location = pvlib.location.Location(
            latitude=df.select("latitude").unique().item(),
            longitude=df.select("longitude").unique().item())
        solar_df = location.get_solarposition(
            dfp['time'] + pd.Timedelta(minutes=30),
            temperature=dfp['airTemperature']).reset_index()
        dni = pvlib.irradiance.disc(
            ghi=dfp["GHI"],
            solar_zenith=solar_df['apparent_zenith'],
            datetime_or_doy=solar_df['time'].dt.dayofyear)
        rad_df = pvlib.irradiance.complete_irradiance(
            solar_zenith=solar_df['apparent_zenith'],
            ghi=dfp["GHI"],
            dni=dni["dni"],
            dhi=None).rename(columns={'ghi': 'GHI', 'dni': 'DNI', 'dhi': 'DHI'})
        solar_df = solar_df.drop(['apparent_zenith', 'zenith', 'apparent_elevation', 'equation_of_time'], axis=1)
        solar_df = solar_df.rename(columns={'elevation': 'sunElevation', 'azimuth': 'sunAzimuth'})
        solar_df['time'] = solar_df['time'] - pd.Timedelta(minutes=30)
        solar_df = pl.from_pandas(pd.concat([solar_df, rad_df], axis=1))

        return solar_df.join(df.drop("GHI"), on="time", how="inner")

    # Loop through each group and apply the join_solar_data function
    result_list = []
    for group_name, group_df in df.group_by("weatherStation"):
        result = join_solar_data(group_df)
        result_list.append(result)
    df = pl.concat(result_list)

    # Dropping latitude and longitude already found in the weatherStation column
    df = df.drop(["latitude", "longitude"])

    return df


def MeteoGalicia_real_time(pred_time, data_dir, lat_range, lon_range):
    # Function that downloads data from the MeteoGalicia dataset in real time : from the prediction day to 4 days after
    # The data is then stored in directories {data_dir} chosen by the user

    year = pred_time.year
    month = pred_time.month
    day = pred_time.day

    next_time = pred_time + datetime.timedelta(days=4)
    next_year = next_time.year
    next_month = next_time.month
    next_day = next_time.day

    north = max(lat_range)
    south = min(lat_range)
    east = max(lon_range)
    west = min(lon_range)

    rest_url = (f"https://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d02/{year}/{month:02}/wrf_arw_det_history_"
                f"d02_{year}{month:02}{day:02}_0000.nc4?var=lat&var=lon&var=prec&var=rh&var=swflx&var=temp&var=u&var=v&"
                f"north={north}&west={west}&east={east}&south={south}&disableProjSubset=on&horizStride=1&"
                f"time_start={year}-{month:02}-{day:02}T01%3A00%3A00Z&"
                f"time_end={next_year}-{next_month:02}-{next_day:02}T00%3A00%3A00Z&timeStride=1&"
                f"accept=netcdf")
    nc_name = f"{year}{month:02}{day:02}_meteogalicia_realtime_96_hours.nc"  # name of download nc file
    nc_file = f"{data_dir}/raw_nc_files/{nc_name}"
    os.makedirs(f"{data_dir}/raw_nc_files", exist_ok=True)
    retour = requests.get(rest_url)
    with open(nc_file, 'wb') as file:
        file.write(retour.content)
    print(f"Data from {day:02}/{month:02}/{year} to {next_day:02}/{next_month:02}/{next_year} has been downloaded successfully!")

    return nc_file


def ERA5Land_historical(ym, data_dir, lat_range, lon_range):
    # Connect to the Copernicus Climate Date Store
    client = None

    def load_copernicus():
        try:
            client = cdsapi.Client()  # You'll need a CDS credentials file in ~/.cdsapirc
        except:
            raise sys.stderr.write(
                "\tObtain credentials to Climate Data Store and set them in ~/.cdsapirc file. "
                "\n\tPlease, follow instructions in: "
                "\n\thttps://cds.climate.copernicus.eu/how-to-api")
        return client

    # Gather the GRIB file for each year-month.
    year = int(str(ym)[0:4])
    month = int(str(ym)[4:6])
    filename = f'{data_dir}/{year}{month:02}_{max(lat_range)}_{min(lon_range)}_{min(lat_range)}_{max(lon_range)}.grib'
    if not os.path.exists(filename):
        if client is None:
            client = load_copernicus()
        dataset = "reanalysis-era5-land"
        request = {
                'variable': [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_dewpoint_temperature",
                    "2m_temperature",
                    "leaf_area_index_high_vegetation",
                    "leaf_area_index_low_vegetation",
                    "total_precipitation",
                    "surface_solar_radiation_downwards",
                    "forecast_albedo",
                    "soil_temperature_level_4",
                    "volumetric_soil_water_layer_4"
                ],
                'year': f'{year}',
                'month': f'{month:02}',
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'data_format': 'grib',
                'download_format': 'unarchived',
                'area': [  # Bounding box
                    max(lat_range), min(lon_range),  # Upper left point (Lat-lon)
                    min(lat_range), max(lon_range)  # Lower right point (Lat-lon)
                ]
            }
        client.retrieve(dataset, request).download(filename)
    print(f"\nData for {month:02}/{year} has been downloaded successfully!")
    return filename


def get_min_max_lat_lon(model_folder):
    high_res_min = np.load(f'{model_folder}/high_min.npy', allow_pickle=True)
    high_res_max = np.load(f'{model_folder}/high_max.npy', allow_pickle=True)
    low_res_min = np.load(f'{model_folder}/low_min.npy', allow_pickle=True)
    low_res_max = np.load(f'{model_folder}/low_max.npy', allow_pickle=True)
    xylatlon = np.load(f'{model_folder}/xylatlon.npy', allow_pickle=True)
    return high_res_min, high_res_max, low_res_min, low_res_max, xylatlon


def format_data(file_dir, low_res_min, low_res_max, static_features_zarr_file, barcelona_shp_dir, catalonia_shp_dir,
                fore, low_res_bbox_polygon=None):
    if fore:
        dataset = xr.open_dataset(file_dir)
        df_ = [
            pl.from_pandas(dataset[var].to_dataframe().reset_index())
            for var in dataset.data_vars
            if dataset[var].ndim > 0
        ]  # getting rid of one dimensional variables
        lat_lon = df_[0].join(df_[1], on=['x', 'y'], how='left')
        # Join all the individual datasets on time, x and y values
        df_ = reduce(lambda left, right: left.join(right, on=['x', 'y', 'time'], how='inner'), df_[2:])
        # Join lat and lon values that correspond to the x and y columns
        df_ = df_.join(lat_lon, on=['x', 'y'], how='left')
        df_ = df_.drop(['x', 'y'])
        # Apply transformation pipeline
        df_ = df_.pipe(MeteoGalicia_transformation)
    else:
        # Opening and converting grib file to dataframe
        hash = random.getrandbits(128)
        dg = cfgrib.open_datasets(file_dir, backend_kwargs={'indexpath': f'{file_dir}.{hash}.idx'})
        os.remove(f'{file_dir}.{hash}.idx')
        df_ = [pl.from_pandas(dg[j].to_dataframe().reset_index()) for j in range(len(dg))]
        # Join all the individual datasets on lat, lon, time triplets
        df_ = reduce(lambda x, y: x.join(y, on=["latitude", "longitude", "valid_time"], how="left"), df_)
        # Apply transformation pipeline
        df_ = df_.pipe(ERA5Land_transformation)

    # Convert back to xarray dataframe
    df_p = df_.to_pandas()
    dataset = xr.Dataset.from_dataframe(df_p.set_index(['time', 'weatherStation']))

    # Filter out the unnecessary weather stations (out of scope)
    ## Filter out the unnecessary weather stations (out of scope)
    if low_res_bbox_polygon is not None:
        input_realtime_ds = filter_weather_station(dataset, low_res_bbox_polygon, "EPSG:4326")
    else:
        barcelona_area = get_gdf_buffed_area(barcelona_shp_dir, catalonia_shp_dir, 5000)
        low_res_bbox_polygon = barcelona_area.values.union_all()
        input_realtime_ds = filter_weather_station(dataset, low_res_bbox_polygon, "EPSG:4326")

    if fore:
        dropped_vars = ["windSpeedEast", "windSpeedNorth"]
        order = ['DHI', 'DNI', 'GHI', 'airTemperature', 'relativeHumidity', 'sunAzimuth', 'sunElevation',
                 'totalPrecipitation', 'windDirection', 'windSpeed']  # var in the same order as model training input
    else:
        dropped_vars = ["dewAirTemperature", "windSpeedEast", "windSpeedNorth"]
        order = ['DHI', 'DNI', 'GHI', 'airTemperature', 'albedo', 'highVegetationRatio', 'lowVegetationRatio',
                 'relativeHumidity', 'soilTemperature', 'soilWaterRatio', 'sunAzimuth', 'sunElevation',
                 'totalPrecipitation', 'windDirection', 'windSpeed']  # var in the same order as model training input
    input_realtime_ds = input_realtime_ds.drop_vars(dropped_vars)
    input_realtime_ds = xr.Dataset({var: input_realtime_ds[var] for var in order})
    # normalize input data from previously saved min and max of the model input data
    input_realtime_ds = minmax_norm_realtime(input_realtime_ds, low_res_min, low_res_max)

    if static_features_zarr_file.endswith(".zip"):
        unzip_zarr_file(static_features_zarr_file)
    static_features_ds = xr.open_zarr(static_features_zarr_file, chunks={})

    input_realtime_ds = format_input_ds(input_realtime_ds, static_features_ds)

    return input_realtime_ds


def add_loc_time(df, nh):
    # Adding time and location data to model input
    df = df.reset_index()
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df[['latitude', 'longitude']] = df['weatherStation'].str.split('_', n=1, expand=True)
    df['latitude'] = df.latitude.astype(float)
    df['longitude'] = df.longitude.astype(float)

    def fs(df, nhar, var, period):
        for h in [i + 1 for i in range(nhar)]:
            aux = 2 * pi * h * df[var] / period
            df[f"{var}_{h}_sin"] = np.sin(aux)
            df[f"{var}_{h}_cos"] = np.cos(aux)
        return df

    df = fs(df, nh, 'hour', 24)
    df = fs(df, nh, 'month', 12)
    df = df.drop(columns=['hour', 'month'])
    df.set_index(['time', 'weatherStation'], inplace=True)
    return df


def general_prediction(pred_time, path_model, data_dir, barcelona_shp_dir, catalonia_shp_dir,
                       static_features_zarr_file, lat_range, lon_range, n_harm, fore, low_res_bbox_polygon):
    # Download realtime data from MeteoGalicia
    if fore:
        file_dir = MeteoGalicia_real_time(pred_time, data_dir, lat_range, lon_range)
    else:
        file_dir = ERA5Land_historical(pred_time, data_dir, lat_range, lon_range)

    high_res_min, high_res_max, low_res_min, low_res_max, xylatlon = get_min_max_lat_lon(data_dir)

    # Format data
    input_realtime_ds = format_data(
        file_dir, low_res_min, low_res_max, static_features_zarr_file, barcelona_shp_dir, catalonia_shp_dir, fore,
        low_res_bbox_polygon
    )

    #os.remove(file_dir)  # Remove nc file or grib file with raw data

    time_steps = input_realtime_ds.coords["time"].values
    X_realtime = input_realtime_ds.to_dataframe()
    X_realtime = add_loc_time(X_realtime, nh=n_harm)
    realtimespace_pool = Pool(data=X_realtime)

    # Load model and predict weather data
    print(f"Model path: {path_model}")
    if path_model[-3:] == 'cbm':
        model = CatBoostRegressor()
        model.load_model(path_model)
        preds = model.predict(realtimespace_pool)
    else:
        print(f"Error: model must be cbm file")
        sys.exit()

    return preds, time_steps


def parse_station_coords(station_ids):
    """Extract latitude and longitude from station ID strings like '41.2_2.1'"""
    lats, lons = zip(*[map(float, s.split('_')) for s in station_ids])
    return np.array(lats), np.array(lons)

def plot_datasets_on_map(low_res_ds, high_res_ds, variable, time_to_plot,
                         barcelona_shp_dir,
                         low_marker_size=80, high_marker_size=30, cmap='viridis',
                         save_path=None):
    """
    Plot low- and high-resolution datasets on a map at a given time, and overlay Barcelona shapefile.

    Parameters:
    - barcelona_shp_dir: directory containing the Barcelona shapefile (.shp and others)
    """
    # Select nearest data at the given time
    low_res = low_res_ds.sel(time=time_to_plot, method='nearest')
    high_res = high_res_ds.sel(time=time_to_plot, method='nearest')

    # Extract coordinates from weatherStation index
    lat_low, lon_low = parse_station_coords(low_res['weatherStation'].values)
    lat_high, lon_high = parse_station_coords(high_res['weatherStation'].values)

    # Get variable values
    val_low = low_res[variable].values
    val_high = high_res[variable].values

    # Load Barcelona shapefile
    barcelona_gdf = gpd.read_file(barcelona_shp_dir)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title(f"{variable} at {np.datetime_as_string(low_res['time'].values, unit='h')}", fontsize=14)

    # Base map
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.gridlines(draw_labels=True)

    # Overlay Barcelona shapefile
    barcelona_gdf = barcelona_gdf.to_crs(epsg=4326)  # Ensure it's in lat/lon
    for geometry in barcelona_gdf.geometry:
        ax.add_geometries([geometry], crs=ccrs.PlateCarree(),
                          facecolor='none', edgecolor='red', linewidth=1.5)

    # High-res points
    sc_high = ax.scatter(lon_high, lat_high, c=val_high, cmap=cmap, s=high_marker_size,
                         edgecolor='k', linewidth=0.2, transform=ccrs.PlateCarree(), label='High-res')

    # Low-res points
    sc_low = ax.scatter(lon_low, lat_low, c=val_low, cmap=cmap, s=low_marker_size,
                        edgecolor='black', linewidth=0.5, transform=ccrs.PlateCarree(), marker='o', label='Low-res')

    # Colorbar
    cbar = plt.colorbar(sc_low, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
    cbar.set_label(variable)

    # Legend
    plt.legend(loc='lower left')

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close(fig)

def plot_air_temperature_raster(df: pl.DataFrame, time_str: str, max_area: float, pdf_path: str):
    """
    Plot and save a raster (2D heatmap) of air temperature for a specific timestamp.

    Parameters:
    - df (pl.DataFrame): Polars DataFrame with 'time', 'latitude', 'longitude', 'airTemperature'
    - time_str (str): Timestamp in 'YYYY-MM-DD HH:MM:SS' format
    - max_area (float): Maximum allowed area per triangle in the triangulation
    - pdf_path (str): Path to save the output PDF (e.g., 'output/plot.pdf')
    """
    # Filter data for the specified time
    df_time = df.filter(
        pl.col("time") == pl.lit(time_str).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    )

    if df_time.is_empty():
        print(f"No data found for time: {time_str}")
        return

    # Ensure lat/lon are floats
    df_time = df_time.with_columns([
        pl.col("latitude").cast(pl.Float64),
        pl.col("longitude").cast(pl.Float64)
    ])

    pdf = df_time.select(["latitude", "longitude", "airTemperature"]).to_pandas()

    lat = pdf["latitude"].values
    lon = pdf["longitude"].values
    temp = pdf["airTemperature"].values

    triang = Triangulation(lon, lat)

    # Compute triangle areas
    x = lon[triang.triangles]
    y = lat[triang.triangles]
    area = 0.5 * np.abs(
        x[:, 0] * (y[:, 1] - y[:, 2]) +
        x[:, 1] * (y[:, 2] - y[:, 0]) +
        x[:, 2] * (y[:, 0] - y[:, 1])
    )

    # Mask large triangles
    mask = area > max_area
    triang.set_mask(mask)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    tpc = ax.tripcolor(triang, temp, shading='flat', cmap='coolwarm')
    plt.colorbar(tpc, label="Air Temperature (°C)")
    ax.set_title(f"Air Temperature (triangles ≤ {max_area} area)\nat {time_str}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    # Save to PDF
    if '/' in pdf_path:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    plt.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"Saved plot to: {pdf_path}")

def plot_humidity_raster(df: pl.DataFrame, time_str: str, max_area: float, pdf_path: str):
    """
    Plot and save a raster (2D heatmap) of relative humidity for a specific timestamp.

    Parameters:
    - df (pl.DataFrame): Polars DataFrame with 'time', 'latitude', 'longitude', 'relativeHumidity'
    - time_str (str): Timestamp in 'YYYY-MM-DD HH:MM:SS' format
    - max_area (float): Maximum allowed area per triangle in the triangulation
    - pdf_path (str): Path to save the output PDF (e.g., 'output/plot.pdf')
    """
    # Filter data for the specified time
    df_time = df.filter(
        pl.col("time") == pl.lit(time_str).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    )

    if df_time.is_empty():
        print(f"No data found for time: {time_str}")
        return

    # Ensure lat/lon are floats
    df_time = df_time.with_columns([
        pl.col("latitude").cast(pl.Float64),
        pl.col("longitude").cast(pl.Float64)
    ])

    pdf = df_time.select(["latitude", "longitude", "relativeHumidity"]).to_pandas()

    lat = pdf["latitude"].values
    lon = pdf["longitude"].values
    rh = pdf["relativeHumidity"].values * 100

    triang = Triangulation(lon, lat)

    # Compute triangle areas
    x = lon[triang.triangles]
    y = lat[triang.triangles]
    area = 0.5 * np.abs(
        x[:, 0] * (y[:, 1] - y[:, 2]) +
        x[:, 1] * (y[:, 2] - y[:, 0]) +
        x[:, 2] * (y[:, 0] - y[:, 1])
    )

    # Mask large triangles
    mask = area > max_area
    triang.set_mask(mask)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    tpc = ax.tripcolor(triang, rh, shading='flat', cmap='coolwarm')
    plt.colorbar(tpc, label="Relative humidity (%)")
    ax.set_title(f"Relative humidity (triangles ≤ {max_area} area)\nat {time_str}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    # Save to PDF
    if '/' in pdf_path:
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    plt.savefig(pdf_path, format='pdf')
    plt.close(fig)
    print(f"Saved plot to: {pdf_path}")

def plot_time_series_bundle(
        df: pl.DataFrame,
        variables: list[str],
        output_file: str = "weather_timeseries.html",
        limits: dict[str, dict] = None,
        group_by_column: str | None = None,
        title_column: str | None = None,
        title: str = "",
        y_limits_axis: tuple[float, float] | None = None
):
    if "time" not in df.columns or not variables:
        raise ValueError("The DataFrame must contain a 'time' column and at least one variable to plot.")

    limits = limits or {}
    group_by_column = group_by_column or ""
    title_column = title_column or ""

    html_chunks = []

    if group_by_column:
        group_values = df.select(group_by_column).unique().to_series().to_list()
        for group_value in group_values:
            df_group = df.filter(pl.col(group_by_column) == group_value)
            title_str = f"{title} - {group_value}" if title else str(group_value)
            html_chunks.append(
                plot_single_time_series(
                    df_group, variables, title_str, limits, title_column, y_limits_axis, as_html=True
                )
            )
    else:
        html_chunks.append(
            plot_single_time_series(df, variables, title, limits, title_column, y_limits_axis, as_html=True)
        )

    # Combine all HTML chunks into one HTML file
    full_html = """
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
    """
    full_html += "\n<hr style='margin:30px 0;'>\n".join(html_chunks)
    full_html += "\n</body></html>"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"✅ All plots saved in: {output_file}")


def plot_single_time_series(
        df: pl.DataFrame,
        variables: list[str],
        title: str,
        limits: dict[str, dict],
        title_column: str | None,
        y_limits_axis: tuple[float, float] | None,
        as_html: bool = False
):
    df_pd = df.select(["time"] + variables + ([title_column] if title_column else [])).to_pandas()
    fig = go.Figure()
    primary_var = variables[0]

    # Primary variable
    fig.add_trace(go.Scatter(
        x=df_pd["time"],
        y=df_pd[primary_var],
        mode="lines",
        name=primary_var,
        line=dict(color="blue")
    ))

    if primary_var in limits:
        limit = limits[primary_var]
        cond = df_pd[primary_var] > limit["value"] if limit["condition"] == "above" else df_pd[primary_var] < limit["value"]
        fig.add_trace(go.Scatter(
            x=df_pd["time"][cond],
            y=df_pd[primary_var][cond],
            mode="markers",
            name=f"{primary_var} {limit['condition']} {limit['value']}",
            marker=dict(color="red", size=6, symbol="circle"),
            showlegend=True
        ))

    # Additional variables
    colors = ["green", "orange", "purple", "brown", "black"]
    for i, var in enumerate(variables[1:], start=1):
        color = colors[(i - 1) % len(colors)]
        fig.add_trace(go.Scatter(
            x=df_pd["time"],
            y=df_pd[var],
            mode="lines",
            name=var,
            yaxis="y",
            line=dict(color=color)
        ))

        if var in limits:
            limit = limits[var]
            cond = df_pd[var] > limit["value"] if limit["condition"] == "above" else df_pd[var] < limit["value"]
            fig.add_trace(go.Scatter(
                x=df_pd["time"][cond],
                y=df_pd[var][cond],
                mode="markers",
                name=f"{var} {limit['condition']} {limit['value']}",
                yaxis="y",
                marker=dict(color="red", size=6, symbol="x"),
                showlegend=True
            ))

    yaxis_config = dict(
        title=primary_var,
        titlefont=dict(color="blue"),
        tickfont=dict(color="blue")
    )

    if y_limits_axis:
        yaxis_config["range"] = list(y_limits_axis)

    fig.update_layout(
        title=title or "Time Series Plot",
        xaxis_title="Time",
        yaxis=yaxis_config,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=60, t=60, b=40),
        height=500,
        template="plotly_white"
    )

    if as_html:
        return plot(fig, include_plotlyjs=False, output_type='div')
    else:
        return fig