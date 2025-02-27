
from catboost import CatBoostRegressor, Pool
import requests
import os
import sys
import xarray as xr
from functools import reduce
import polars as pl
import numpy as np
import pvlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import datetime
from math import pi
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import zipfile
import cdsapi
import cfgrib
import random
import warnings


def get_gdf_buffed_area(barcelona_shapefile_directory, catalonia_shapefile_directory, distance):
    # Distance in meters /!\
    gdf_b = gpd.read_file(barcelona_shapefile_directory)
    gdf_c = gpd.read_file(catalonia_shapefile_directory)
    crs_b = gdf_b.crs
    # Make sure the coordinate reference system is the same
    if crs_b != gdf_c.crs:
        gdf_c = gdf_c.to_crs(crs_b)
    # Union on multipolygons
    barcelona_polygon = gdf_b.union_all()
    catalonia_polygon = gdf_c.union_all()
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
                barcelona_shp_dir, catalonia_shp_dir):
    ## Unzipping data if not unzipped
    for file in [high_res_zarr_dir, low_res_hist_zarr_dir, low_res_fore_zarr_dir, static_features_zarr_file]:
        unzip_zarr_file(file)

    ## Open data
    high_res_ds = xr.open_zarr(high_res_zarr_dir, chunks={})
    low_res_hist_ds = xr.open_zarr(low_res_hist_zarr_dir, chunks={})
    low_res_fore_ds = xr.open_zarr(low_res_fore_zarr_dir, chunks={})
    static_features_ds = xr.open_zarr(static_features_zarr_file, chunks={})

    ## Filter out the unnecessary weather stations (out of scope)
    barcelona_area = get_gdf_buffed_area(barcelona_shp_dir, catalonia_shp_dir, 10000)
    crs_b = barcelona_area.crs
    barcelona_buffered_polygon = barcelona_area.union_all()
    gdf_b = gpd.read_file(barcelona_shp_dir)
    barcelona_regular_polygon = gdf_b.union_all()

    high_res_ds = filter_weather_station(high_res_ds, barcelona_regular_polygon, crs_b)
    low_res_fore_ds = filter_weather_station(low_res_fore_ds, barcelona_buffered_polygon, crs_b)
    low_res_hist_ds = filter_weather_station(low_res_hist_ds, barcelona_buffered_polygon, crs_b)

    return high_res_ds, low_res_fore_ds, low_res_hist_ds, static_features_ds


def minmax_normalization(ds):
    max_vars = {}
    min_vars = {}
    for var in ds.data_vars:
        if var != 'x' and var != 'y':
            print(f'Normalizing {var}')
            max = ds[var].max(dim="time")
            min = ds[var].min(dim="time")
            max_vars[var] = max
            min_vars[var] = min
            ds[var] = (ds[var] - min) / (max - min)  # On part du principe que max et min sont différents
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

    rest_url = f"https://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d02/{year}/{month:02}/wrf_arw_det_history_d02_{year}{month:02}{day:02}_0000.nc4?var=lat&var=lon&var=prec&var=rh&var=swflx&var=temp&var=u&var=v&north={north}&west={west}&east={east}&south={south}&disableProjSubset=on&horizStride=1&time_start={year}-{month:02}-{day:02}T01%3A00%3A00Z&time_end={next_year}-{next_month:02}-{next_day:02}T00%3A00%3A00Z&timeStride=1&accept=netcdf"
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
    print(f"Data for {month:02}/{year} has been downloaded successfully!")
    return filename


def get_min_max_lat_lon(model_folder):
    high_res_min = np.load(f'{model_folder}/highmin.npy', allow_pickle=True)
    high_res_max = np.load(f'{model_folder}/highmax.npy', allow_pickle=True)
    low_res_min = np.load(f'{model_folder}/lowmin.npy', allow_pickle=True)
    low_res_max = np.load(f'{model_folder}/lowmax.npy', allow_pickle=True)
    xylatlon = np.load(f'{model_folder}/xylatlon.npy', allow_pickle=True)
    return high_res_min, high_res_max, low_res_min, low_res_max, xylatlon


def format_data(file_dir, low_res_min, low_res_max, static_features_zarr_file, barcelona_shp_dir, catalonia_shp_dir, fore):
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
    barcelona_area = get_gdf_buffed_area(barcelona_shp_dir, catalonia_shp_dir, 10000)
    crs_b = barcelona_area.crs
    barcelona_buffered_polygon = barcelona_area.union_all()
    input_realtime_ds = filter_weather_station(dataset, barcelona_buffered_polygon, crs_b)

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
    df['longitude'] = df.latitude.astype(float)

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
                       static_features_zarr_file, lat_range, lon_range, n_harm, fore):
    # Download realtime data from MeteoGalicia
    if fore:
        file_dir = MeteoGalicia_real_time(pred_time, data_dir, lat_range, lon_range)
    else:
        file_dir = ERA5Land_historical(pred_time, data_dir, lat_range, lon_range)

    high_res_min, high_res_max, low_res_min, low_res_max, xylatlon = get_min_max_lat_lon(data_dir)

    # Format data
    input_realtime_ds = format_data(
        file_dir, low_res_min, low_res_max, static_features_zarr_file, barcelona_shp_dir, catalonia_shp_dir, fore
    )

    os.remove(file_dir)  # Remove nc file or grib file with raw data

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

