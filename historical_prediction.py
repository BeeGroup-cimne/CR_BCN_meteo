import os
from utils import *
from shapely.geometry import Polygon

if __name__ == "__main__":
    # download weather data from era5land website
    # format raw data
    # load model to be used for prediction
    # predict values
    # save as csv file with columns: prediction time, time, weather station, temperature, relative humidity

    # General parameters
    ym_range = [201901, 202505]  # range that is going to be predicted formatted like YYYYMMDD
    model_name = "cat_0_41.2,41.6_1.9,2.4_26280_200_4_400_0.03_6_4"
    model_extension = '.cbm'
    lat_range = [float(model_name.split("_")[2].split(",")[0]), float(model_name.split("_")[2].split(",")[1])]
    lon_range = [float(model_name.split("_")[3].split(",")[0]), float(model_name.split("_")[3].split(",")[1])]
    k_fore = int(model_name.split("_")[1])
    N_hours = int(model_name.split("_")[4])
    n_steps = int(model_name.split("_")[5])
    n_harmonics = int(model_name.split("_")[6])
    iterations = int(model_name.split("_")[7])
    learn_rate = float(model_name.split("_")[8])
    depth = int(model_name.split("_")[9])
    min_weight = int(model_name.split("_")[10])

    model_file = model_name + model_extension
    nextcloud_root_dir = os.path.expanduser('~/Nextcloud2/Beegroup/data/CR_BCN_meteo')
    static_features_zarr_file = f'{nextcloud_root_dir}/General_Data/weather_static_features.zarr'
    barcelona_shp_dir = f'{nextcloud_root_dir}/General_Data/shapefiles_barcelona_distrito.shp'
    catalonia_shp_dir = f'{nextcloud_root_dir}/General_Data/divisions-administratives-v2r1-catalunya-5000-20240705.shp'
    path_model = f'{nextcloud_root_dir}/Historical_ERA5Land/{model_file}'
    os.makedirs(f'{nextcloud_root_dir}/Historical_ERA5Land/Predictions', exist_ok=True)

    val_names = ['airTemperature', 'relativeHumidity']  # variable names in the model output

    ym = min(ym_range)
    year, month = int(str(ym)[:4]), int(str(ym)[4:6])
    df_final = None
    while ym <= max(ym_range):  # data loaded for range of days chosen by user
        final_file_name = f'prediction_{ym}'
        file_path = f'{nextcloud_root_dir}/Historical_ERA5Land/Predictions/{final_file_name}.parquet'
        if not os.path.exists(file_path):
            prediction, time_steps = general_prediction(
                ym, path_model, f'{nextcloud_root_dir}/Historical_ERA5Land',
                barcelona_shp_dir, catalonia_shp_dir, static_features_zarr_file,
                lat_range, lon_range, n_harmonics, fore=k_fore,
                low_res_bbox_polygon = Polygon([
                    (lon_range[0], lat_range[0]),  # bottom-left
                    (lon_range[1], lat_range[0]),  # bottom-right
                    (lon_range[1], lat_range[1]),  # top-right
                    (lon_range[0], lat_range[1]),  # top-left
                    (lon_range[0], lat_range[0])   # back to bottom-left to close the polygon
                ]))
            nspace = 10222
            ntime = int(prediction.shape[0] / nspace)
            high_res_min, high_res_max, low_res_min, low_res_max, xylatlon = get_min_max_lat_lon(f'{nextcloud_root_dir}/Historical_ERA5Land')
            df_time = None
            for ival, name in enumerate(val_names):
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
            print(f'Predictions successfully saved in parquet file and uploaded to NextCloud')
            df_time.write_parquet(file_path)
        month += 1
        if month > 12:
            year += 1
            month = 1
        ym = int(f'{year}{month:02}')
