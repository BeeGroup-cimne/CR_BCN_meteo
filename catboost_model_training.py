
from utils import *
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import shuffle
from catboost import CatBoostRegressor, Pool
from math import ceil


if __name__ == "__main__":
    # Parameters
    k_fore = 0  # Boolean set to 1 when training MeteoGalicia (forecasting) model, and 0 for ERA5Land (historical) model
    model_name = 'cat_hist_8yrs_04_6_10_timespace'  # choose recognizable name (with hyperparameters used for example)
    model_extension = '.cbm'
    N_hours = int(8 * 8760)  # Number of hours (scenarios) kept
    n_steps = 20 * 8  # number of chunks, increase if sigkill signal. In general for ERA5Land 20*8, MeteoGalicia 8*8
    n_harmonics = 4  # number of harmonics when using Fourier series to represent time input
    # Model hyperparameters
    iterations = 100
    learn_rate = 0.04
    depth = 6
    min_weight = 10
    # Directories
    nextcloud_root_dir = os.path.expanduser('~/NextCloud/ClimateReady-BCN/WP3-VulnerabilityMap/Weather Downscaling/Models_and_predictions/')

    high_res_zarr_dir = f'{nextcloud_root_dir}General_Data/weather_urbclim_2008-2017.zarr'
    low_res_hist_zarr_dir = f'{nextcloud_root_dir}General_Data/weather_era5land_2008-2017.zarr'
    low_res_fore_zarr_dir = f'{nextcloud_root_dir}General_Data/weather_meteogalicia_2008-2017.zarr'
    static_features_zarr_file = f'{nextcloud_root_dir}General_Data/weather_static_features.zarr'
    barcelona_shp_dir = f'{nextcloud_root_dir}General_Data/shapefiles_barcelona_distrito.shp'
    catalonia_shp_dir = f'{nextcloud_root_dir}General_Data/divisions-administratives-v2r1-catalunya-5000-20240705.shp'
    model_file = model_name + model_extension

    print("Opening and formatting data")
    high_res_ds, low_res_fore_ds, low_res_hist_ds, static_features_ds = get_dataset(
        high_res_zarr_dir,
        low_res_hist_zarr_dir,
        low_res_fore_zarr_dir,
        static_features_zarr_file,
        barcelona_shp_dir,
        catalonia_shp_dir)

    low_res_ds = low_res_fore_ds if k_fore else low_res_hist_ds
    del low_res_fore_ds, low_res_hist_ds
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
        "early_stopping_rounds": 20
    }
    model = CatBoostRegressor(**params)

    for step in range(n_steps):
        print(f'Loop n°{step+1} / {n_steps}')
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
        # Nspacehigh = int(high_res_df_i.shape[0] / n_hours)
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
