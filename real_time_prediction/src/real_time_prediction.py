from utils import *

if __name__ == "__main__":
    # download weather data from meteogalicia website
    # format raw data
    # load model to be used for prediction
    # predict values
    # save as csv file with columns: prediction time, time, weather station, temperature, relative humidity

    # General parameters
    # lat_range = [41.25, 41.6]
    # lon_range = [1.9, 2.35]
    lat_range = [41.25, 41.6]
    lon_range = [1.9, 2.35]
    nd = 1  # number of days to consider BEFORE the prediction time
    n_harm = 4  # Nb of harmonics used for Fourier series, must match with n_harm used for model training
    model_name = 'cat_8yrs_04_8_10_timespace'
    model_extension = '.cbm'

    # nextcloud_root_dir = os.path.expanduser('~/NextCloud/ClimateReady-BCN/WP3-VulnerabilityMap/Weather Downscaling/Models_and_predictions/')

    static_features_zarr_file = f'real_time_prediction/data/weather_static_features.zarr'
    barcelona_shp_dir = f'real_time_prediction/data/shapefiles_barcelona_distrito.shp'
    catalonia_shp_dir = f'real_time_prediction/data/divisions-administratives-v2r1-catalunya-5000-20240705.shp'

    val_names = ['airTemperature', 'relativeHumidity']  # variable names in the model output

    #
    pred_time = datetime.datetime.now()
    final_file_name = f'prediction_{pred_time.year}-{pred_time.month:02}-{pred_time.day:02}'
    model_file = model_name + model_extension
    path_model = f'real_time_prediction/data/{model_file}'
    df_final = None
    for i in range(nd):  # data loaded for nd days before prediction time
        break
        prediction, time_steps = general_prediction(pred_time, path_model, f'real_time_prediction/data/predictions/meteogalicia',
                                                    barcelona_shp_dir, catalonia_shp_dir, static_features_zarr_file,
                                                    lat_range, lon_range, n_harm, fore=1)
        nspace = 10222
        ntime = int(prediction.shape[0] / nspace)
        high_res_min, high_res_max, low_res_min, low_res_max, xylatlon = get_min_max_lat_lon(f'real_time_prediction/data/predictions/meteogalicia')
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
