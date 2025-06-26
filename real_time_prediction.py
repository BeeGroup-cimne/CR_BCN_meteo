from utils import *
from shapely.geometry import Polygon

if __name__ == "__main__":
    # download weather data from meteogalicia website
    # format raw data
    # load model to be used for prediction
    # predict values
    # save as csv file with columns: prediction time, time, weather station, temperature, relative humidity

    # General parameters
    model_name = "cat_1_41.2,41.6_1.9,2.4_26280_64_3_400_0.03_6_4"
    model_extension = '.cbm'
    nd = 3  # number of days to consider BEFORE the prediction time (0 will get from today to the next 4 days)
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

    nextcloud_root_dir = os.path.expanduser('~/Nextcloud2/Beegroup/data/CR_BCN_meteo')
    os.makedirs(f'{nextcloud_root_dir}/General_Data', exist_ok=True)
    os.makedirs(f'{nextcloud_root_dir}/Forecasting_MeteoGalicia', exist_ok=True)
    os.makedirs(f'{nextcloud_root_dir}/Historical_ERA5Land', exist_ok=True)
    plots_dir = f"{nextcloud_root_dir}/Plots_training"
    os.makedirs(plots_dir, exist_ok=True)
    high_res_zarr_dir = f'{nextcloud_root_dir}/General_Data/weather_urbclim_new_2008-2017.zarr'
    low_res_hist_zarr_dir = f'{nextcloud_root_dir}/General_Data/weather_era5land_200801-201712.zarr'
    low_res_fore_zarr_dir = f'{nextcloud_root_dir}/General_Data/weather_meteogalicia_200801-201712.zarr'
    static_features_zarr_file = f'{nextcloud_root_dir}/General_Data/weather_static_features.zarr'
    barcelona_shp_dir = f'{nextcloud_root_dir}/General_Data/shapefiles_barcelona_distrito.shp'
    catalonia_shp_dir = f'{nextcloud_root_dir}/General_Data/divisions-administratives-v2r1-catalunya-5000-20240705.shp'
    model_file = model_name + model_extension

    val_names = ['airTemperature', 'relativeHumidity']  # variable names in the model output

    pred_time = datetime.datetime.now()
    final_file_name = f'prediction_{pred_time.year}-{pred_time.month:02}-{pred_time.day:02}'
    path_model = f'{nextcloud_root_dir}/Forecasting_MeteoGalicia/{model_file}'
    df_final = None
    for i in range(nd+1):  # data loaded for nd days before prediction time
        prediction, time_steps = general_prediction(
            pred_time, path_model, f'{nextcloud_root_dir}/Forecasting_MeteoGalicia',
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
        high_res_min, high_res_max, low_res_min, low_res_max, xylatlon = get_min_max_lat_lon(f'{nextcloud_root_dir}/Forecasting_MeteoGalicia')
        forecasting_time = pred_time.replace(hour=1, minute=0, second=0, microsecond=0)
        forecasting_time = np.datetime64(forecasting_time, 'ns')
        fore_time = np.full(nspace, forecasting_time)
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
                    "forecastingTime": fore_time,
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
                df_time = df_val.join(df_time, on=['forecastingTime', 'time', 'weatherStation'], how='inner')
            else:
                df_time = df_val
        # concatenate dataframes of the different forecasting times into the final dataframe
        if df_final is not None:
            df_final = pl.concat([df_final, df_time])
        else:
            df_final = df_time
        pred_time = pred_time - datetime.timedelta(days=1)

    # save as parquet file the final dataframe
    df_final.write_parquet(f'{nextcloud_root_dir}/Forecasting_MeteoGalicia/Predictions/{final_file_name}.parquet')
    print(f'Predictions successfully saved to NextCloud in parquet file')
