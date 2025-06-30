from utils import *
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import shuffle
from catboost import CatBoostRegressor, Pool
from math import ceil
from tqdm import tqdm
from shapely.geometry import Polygon

if __name__ == "__main__":
    # Parameters
    lat_range = [41.2, 41.6] # [Garraf, Cardedeu]
    lon_range = [1.9, 2.4]
    k_fore = 0 # Boolean set to 1 when training MeteoGalicia (forecasting) model, and 0 for ERA5Land (historical) model
    N_hours = int(3 * 8760)  # Number of hours (scenarios) kept
    n_steps = 25 * 8  # number of chunks, increase if sigkill signal. In general for ERA5Land 20*8, MeteoGalicia 8*8
    n_harmonics = 4  # number of harmonics when using Fourier series to represent time input
    # Model hyperparameters
    iterations = 400
    learn_rate = 0.03
    depth = 6
    min_weight = 4
    model_name = (f'cat_{k_fore}_{",".join([str(i) for i in lat_range])}_{",".join([str(i) for i in lon_range])}_'
                  f'{N_hours}_{n_steps}_{n_harmonics}_{iterations}_{learn_rate}_{depth}_{min_weight}')
    model_extension = '.cbm'
    # Directoriesrow_sums = all_proportion_dfs[variable].sum(axis=1).values[:, None]  # shape (n, 1)
    # all_proportion_dfs[variable] = (all_proportion_dfs[variable].values * 100) / row_sums
    # all_proportion_dfs[variable] = pd.DataFrame(
    #     all_proportion_dfs[variable],
    #     index=all_proportion_dfs[variable].index,
    #     columns=all_proportion_dfs[variable].columns
    # )
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

    print("Opening and formatting data")
    high_res_ds, low_res_fore_ds, low_res_hist_ds, static_features_ds = get_dataset(
        high_res_zarr_dir,
        low_res_hist_zarr_dir,
        low_res_fore_zarr_dir,
        static_features_zarr_file,
        barcelona_shp_dir,
        catalonia_shp_dir,
        hypercadaster_ES_dir = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES",
        low_res_bbox_polygon = Polygon([
            (lon_range[0], lat_range[0]),  # bottom-left
            (lon_range[1], lat_range[0]),  # bottom-right
            (lon_range[1], lat_range[1]),  # top-right
            (lon_range[0], lat_range[1]),  # top-left
            (lon_range[0], lat_range[0])   # back to bottom-left to close the polygon
        ]),
        plots=True,
        plots_dir=plots_dir,
        k_fore=k_fore
    )

    low_res_ds = low_res_fore_ds if k_fore else low_res_hist_ds
    del low_res_fore_ds, low_res_hist_ds

    # Plot the air temperature at some moment
    time_to_plot = "2016-08-20T14:00:00"
    plot_datasets_on_map(
        low_res_ds=low_res_ds,
        high_res_ds=high_res_ds,
        variable='airTemperature',
        barcelona_shp_dir=barcelona_shp_dir,
        low_marker_size=80,
        high_marker_size=2,
        time_to_plot=time_to_plot,
        save_path=f"{plots_dir}/temperature_map_{time_to_plot}_{k_fore}.png",
        prefix_title="ERA5Land and UrbClim " if k_fore==0 else "MeteoGalicia and UrbClim "
    )

    # # Testing weather summarises by year and an approximate location
    # lat = 41.4
    # lon = 2.1
    # hr_ws = static_features_ds["index"].to_dict()["data"]
    # hr_ws = pd.DataFrame([s.split('_') for s in hr_ws], columns=["lat", "lon"])
    # hr_ws["lat"] = hr_ws["lat"].astype(float)
    # hr_ws["lon"] = hr_ws["lon"].astype(float)
    # lr_ws = low_res_ds["weatherStation"].to_dict()["data"]
    # lr_ws = pd.DataFrame([s.split('_') for s in lr_ws], columns=["lat", "lon"])
    # lr_ws["lat"] = lr_ws["lat"].astype(float)
    # lr_ws["lon"] = lr_ws["lon"].astype(float)
    # hr_ws["dist"] = np.sqrt((hr_ws["lat"] - lat) ** 2 + (hr_ws["lon"] - lon) ** 2)
    # lr_ws["dist"] = np.sqrt((lr_ws["lat"] - lat) ** 2 + (lr_ws["lon"] - lon) ** 2)
    # closer_hr_ws = f'{hr_ws.loc[hr_ws["dist"].idxmin(), "lat"]}_{hr_ws.loc[hr_ws["dist"].idxmin(), "lon"]}'
    # closer_lr_ws = f'{lr_ws.loc[lr_ws["dist"].idxmin(), "lat"]}_{lr_ws.loc[lr_ws["dist"].idxmin(), "lon"]}'
    # sf = static_features_ds.to_pandas()
    # hr = high_res_ds.sel(weatherStation=closer_hr_ws).to_pandas()
    # lr = low_res_ds.sel(weatherStation=closer_lr_ws).to_pandas()
    # yearly_degree_days(hr)
    # yearly_degree_days(lr)
    # def yearly_degree_days(df, hdd_base=18, cdd_base=21):
    #     # Daily average temperature
    #     daily_avg = df['airTemperature'].resample('D').mean().to_frame('airTemperature')
    #
    #     # Extract year and date from the index
    #     daily_avg['year'] = daily_avg.index.year
    #     daily_avg['date'] = daily_avg.index.date
    #
    #     # Calculate daily HDD and CDD
    #     daily_avg['HDD'] = (hdd_base - daily_avg['airTemperature']).clip(lower=0)
    #     daily_avg['CDD'] = (daily_avg['airTemperature'] - cdd_base).clip(lower=0)
    #
    #     # Annual sum of HDD and CDD
    #     annual_degree_days = daily_avg.groupby('year')[['HDD', 'CDD']].sum()
    #
    #     return annual_degree_days

    # Filter out missing timestamps between datasets
    common_time = np.intersect1d(high_res_ds.time, low_res_ds.time)
    high_res_ds, low_res_ds = [ds.sel(time=common_time) for ds in [high_res_ds, low_res_ds]]
    n_hours = ceil(N_hours / n_steps)
    time_steps = n_hours * np.arange(n_steps + 1)
    time_steps[-1] = N_hours
    i_times = shuffle(np.arange(N_hours), random_state=42)

    # Fixed validation set
    random_indices = np.random.choice(i_times, size=1000, replace=False)
    high_res_ds_i, low_res_ds_i, high_res_min, high_res_max, low_res_min, low_res_max, high_xy = transformation_general(
        high_res_ds,
        low_res_ds,
        static_features_ds,
        random_indices,
        k_fore
    )

    save_info(high_res_ds, high_res_min, high_res_max, low_res_min, low_res_max,
              nextcloud_local_dir=f'{nextcloud_root_dir}/Forecasting_MeteoGalicia' if k_fore
              else f'{nextcloud_root_dir}/Historical_ERA5Land')

    Y_val, X_val = [ds.to_dataframe() for ds in [high_res_ds_i, low_res_ds_i]]
    X_val = add_loc_time(X_val, nh=n_harmonics)

    val_pool = Pool(data=X_val, label=Y_val)

    params = {
        "iterations": iterations,  # Equivalent to `num_boost_round`
        "learning_rate": learn_rate,
        "depth": depth,  # Equivalent to `max_depth`
        "l2_leaf_reg": min_weight,  # Equivalent to `min_child_weight`
        "loss_function": "MultiRMSE",  # Objective function (multi-regression for multiple urbclim values)
        "eval_metric": "MultiRMSE",
        "verbose": 10,
        "early_stopping_rounds": 30
    }
    model = CatBoostRegressor(**params)

    for step in tqdm(range(n_steps),"Model step:"):
        i_time_chunk = i_times[time_steps[step]: time_steps[step + 1]]
        high_res_ds_i, low_res_ds_i, high_res_min, high_res_max, low_res_min, low_res_max, high_xy = transformation_general(
            high_res_ds,
            low_res_ds,
            static_features_ds,
            i_time_chunk,
            k_fore
        )
        high_res_df_i, low_res_df_i = [ds.to_dataframe() for ds in [high_res_ds_i, low_res_ds_i]]
        del high_res_ds_i, low_res_ds_i
        low_res_df_i = add_loc_time(low_res_df_i, nh=n_harmonics)
        #Nspacehigh = int(high_res_df_i.shape[0] / n_hours)
        Nspacehigh = 10222
        X_train, X_test, Y_train, Y_test, train_times, test_times = data_split(low_res_df_i, high_res_df_i,
                                                                               test_ratio=0.1, nspace=Nspacehigh)
        del low_res_df_i, high_res_df_i
        print('Creating Pools')
        train_pool = Pool(X_train, Y_train)
        test_pool = Pool(X_test, Y_test)
        del X_train, Y_train, X_test
        print('Training model')
        model.fit(
            train_pool,
            eval_set=val_pool,
            init_model=model if step > 0 else None,  # Use the previous model for incremental training
            verbose=params["verbose"],
            early_stopping_rounds=params["early_stopping_rounds"],
            use_best_model=True
        )
        print('Predicting test set')
        preds = model.predict(test_pool)
        rmse = root_mean_squared_error(Y_test, preds)
        print(f"RMSE of the base model (test set): {rmse:.3f}")
    model.save_model(f'{nextcloud_root_dir}/Forecasting_MeteoGalicia/{model_file}' if k_fore
                     else f'{nextcloud_root_dir}/Historical_ERA5Land/{model_file}')
    preds = model.predict(val_pool)
    rmse = root_mean_squared_error(Y_val, preds)
    print(f"RMSE of the base model (validation set -used as eval_set): {rmse:.3f}")
