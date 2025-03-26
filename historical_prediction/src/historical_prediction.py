import numpy as np
import pandas as pd
import polars as pl

from load import harmonize_endesa
from utils import general_prediction, get_min_max_lat_lon


if __name__ == "__main__":
    # download weather data from era5land website
    # format raw data
    # load model to be used for prediction
    # predict values
    # save as csv file with columns: prediction time, time, weather station, temperature, relative humidity

    # General parameters
    # lat_range = [41.25, 41.6]
    # lon_range = [1.9, 2.35]
    # ym_range = [202001, 202002]
    lat_range = [42.9, 40.5]
    lon_range = [0.1, 3.4]
    ym_range = [202201, 202212]  # range that is going to be predicted formatted like YYYYMMDD
    n_harm = 1  # Nb of harmonics used for Fourier series, must match with n_harm used for model training
    model_name = 'cat_hist_8yrs_08_8_10_timespace'
    model_extension = '.cbm'

    # nextcloud_root_dir = os.path.expanduser('~/NextCloud/ClimateReady-BCN/WP3-VulnerabilityMap/Weather Downscaling/Models_and_predictions/')

    static_features_zarr_file = f'historical_prediction/data/weather_static_features.zarr'
    barcelona_shp_dir = f'historical_prediction/data/shapefiles_barcelona_distrito.shp'
    catalonia_shp_dir = f'historical_prediction/data/divisions-administratives-v2r1-catalunya-5000-20240705.shp'

    val_names = ['airTemperature', 'relativeHumidity']  # variable names in the model output

    ym = min(ym_range)
    year, month = int(str(ym)[:4]), int(str(ym)[4:6])
    final_file_name = f'prediction_{ym}-{max(ym_range)}'
    final_file_name
    model_file = model_name + model_extension
    path_model = f'historical_prediction/data/cat_hist_8yrs_08_8_6_timespace.cbm'
    df_final = None
    while ym <= max(ym_range):  # data loaded for range of days chosen by user
        prediction, time_steps = general_prediction(ym, path_model, f'historical_prediction/data/predictions/era5land',
                                                    barcelona_shp_dir, catalonia_shp_dir, static_features_zarr_file,
                                                    lat_range, lon_range, n_harm, fore=0)
        nspace = 10222
        ntime = int(prediction.shape[0] / nspace)
        high_res_min, high_res_max, low_res_min, low_res_max, xylatlon = get_min_max_lat_lon(f'historical_prediction/data/predictions/era5land')
        df_time = None
        for ival, name in enumerate(val_names):
            print(ival, name)
            mmin = high_res_min[:, ival]
            mmax = high_res_max[:, ival]
            weather_stations = xylatlon[:, 0]
            df_val = None
            for itime in range(ntime):
                timedata = prediction[itime * nspace:(itime + 1) * nspace, ival]
                # De-normalizing
                temps = timedata.reshape(timedata.shape[0]) * (mmax - mmin) + mmin  # array of size (Nspace) with data corresponding to timestep itime of variable ival
                time_data = np.full(temps.shape, time_steps[itime])
                df_ = pl.DataFrame({
                    "time": time_data,
                    "weatherStation": weather_stations,
                    f"{name}": temps
                })
                # concatenate the dataframes of every hour
                if df_val is not None:
                    df_val = pl.concat([df_val, df_])
                else:
                    df_val = df_
                df_val = df_val.rechunk()
            # join the two variable (temp and rh) datasets on forecasting time, time, weather station
            if df_time is not None:
                df_time = df_val.join(df_time, on=['time', 'weatherStation'], how='inner')
            else:
                df_time = df_val
        # concatenate dataframes of the different forecasting times into the final dataframe
        if df_final is not None:
            df_final = pl.concat([df_final, df_time])
        else:
            df_final = df_time
        month += 1
        if month > 12:
            year += 1
            month = 1
        ym = int(f'{year}{month:02}')

    df_final_lat_lon = df_final.with_columns([
        df_final["weatherStation"].str.split("_").list.get(0).cast(pl.Float64).alias("latitude"),
        df_final["weatherStation"].str.split("_").list.get(1).cast(pl.Float64).alias("longitude")
    ])
    lat = [41.25, 41.6]
    lon = [1.9, 2.35]

    df_final_bcn = df_final_lat_lon.filter(
        (df_final_lat_lon["latitude"] >= lat[0]) & (df_final_lat_lon["latitude"] <= lat[1]) &
        (df_final_lat_lon["longitude"] >= lon[0]) & (df_final_lat_lon["longitude"] <= lon[1])
    )
    # df_final_bcn.write_csv(f'historical_prediction/data/predictions/{final_file_name}_bcn.csv')
    # save as parquet file the final dataframe
    # df_final.write_parquet(f'historical_prediction/data/predictions/{final_file_name}.parquet')
    # print(f'Predictions successfully saved in parquet file and uploaded to NextCloud')
    # df_original = pd.read_parquet(
    #     '/Users/jose/Nextcloud/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Weather Downscaling/Models_and_predictions/Historical_ERA5Land/Predictions/prediction_202006-202007.parquet')
    # print(df_original.head())
    df_final_bcn = pl.read_csv("historical_prediction/data/predictions/prediction_202201-202212.csv")
    df_final_bcn = df_final_bcn.with_columns(pl.col("time").str.to_datetime(time_unit="ns").alias("time"))
    df_filtered = df_final_bcn.filter(pl.col("time").dt.month().is_between(9, 9))
    harmonize_endesa(df_filtered.to_pandas())
