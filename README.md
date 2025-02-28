# CR_BCN_meteo
You will find in this repo python scripts that train weather downscaling 
models (catboost_model_training.py) and can predict high-precision 
weather data from said models (historical_prediction.py & real_time_prediction.py).
They download and upload files stored in the BEE Group's Next Cloud.

## Authors
- Anceline Desertaux - anceline.desertaux@insa-lyon.fr
- Arnau Comas
- Jose Manuel Broto - jmbroto@cimne.upc.edu
- Gerard Mor - gmor@cimne.upc.edu

## catboost_model_training.py
Script training a weather downscaling model, either for 
MeteoGalicia data (forecasting model) or for ERA5-Land data 
(historical model).

### Parameters
The parameters that can be changed are listed at the beginning 
of the script, they are:
- k_fore: Boolean set to 1 for MeteoGalicia model, and 0 for ERA5Land model
- model_name and model_extension : generally .cbm extension, choose a logical 
and recognizable name
- N_hours: Nb of hours kept from the datasets, usually set to 8 * 8760
- n_steps: Nb of training chunks, generally set to 20 * 8 for ERA5Land 
and 8 * 8 for MeteoGalicia
- n_harmonics: Nb of harmonics in the Fourier series for addition of 
the hour and month to the input dataset. Generally set to 4
- Model hyperparameters (iterations, learn_rate, depth, min_weight):
to choose according to the model that is to be created
- Directories: nextcloud_root_dir (local directory to next cloud's 
weather downscaling folder) and other file names that are found in 
Next Cloud (their names don't have to be changed unless they change 
in Next Cloud)

### Next Cloud directories
An important point should be made on the different Next Cloud directories 
considered here. The root folder should correspond to:
'ClimateReady-BCN/WP3-VulnerabilityMap/Weather Downscaling/Models_and_predictions'.
Adjust the script so that it corresponds to your path to this Next Cloud 
folder. After that, within this root folder, there are 3 folders:
- Forecasting_MeteoGalicia: containing the saved items after training 
a MeteoGalicia model, and a 'Predictions' folder containing predictions 
made with a MeteoGalicia model
- Historical_ERA5Land: containing the saved items after training 
an ERA5Land model, and a 'Predictions' folder containing predictions 
made with a ERA5Land model
- General_Data: containing data useful to train both types of model, 
meaning all the datasets as zipped .zarr files (static input, dynamic 
MeteoGalicia input, dynamic ERA5Land input, dynamic UrbClim output) 
and .shp files of Catalonia and Barcelona to filter out the target 
locations

### Loading and saving items
The script loads the files in the General_Data folder at the beginning.

It then trains the chosen model sequentially, with the CatBoost library.

And along the way saves several pieces of information that are to be reused 
when wanting to make predictions once the model is trained. They are 
stored in the 'Forecasting_MeteoGalicia' folder or the 'Historical_ERA5Land'
folder depending on the value of 'k_fore'. The items saved are:
- the model itself
- 5 .npy files ('high_min', 'high_max', 'low_min', 'low_max', 'xylatlon')
that store information about the maximums and minimums for a future 
normalization of input data and de-normalization of output data, along with
the spatial information of the input dataset for the reference to the 
weather stations

### Models trained so far (Research paper Anceline Desertaux)
Note that grid search doesn't work as wanted with sequential training,
the best-fitting values for some of the model's hyperparameters have 
thus to be determined with a manual tuning, a method is presented in 
Anceline's paper. Historical and forecasting weather downscaling 
models with the right hyperparameters have already been determined and 
are stored in Next Cloud. The parameters that have been used are the 
following:
- N_hours = int(8 * 8760)
- n_steps = 20 * 8 for ERA5Land data or 8 * 8 for MeteoGalicia data
- n_harmonics = 4
- iterations = 100

And for each model, the hyperparameters showing the best performance were:
- Forecasting (MeteoGalicia) : learn_rate = 0.04, depth = 6, min_weight = 8
- Historial (ERA5Land) : learn_rate = 0.08, depth = 8, min_weight = 6

They are named 'cat_8yrs_04_6_8_timespace.cbm' and 
'cat_hist_8yrs_08_8_6_timespace.cbm'. For models named otherwise, the 
value of the parameters are bound to change.

## historical_prediction.py
Script predicting high-resolution weather data on Barcelona from historical 
data from ERA5Land, using a trained weather downscaling model (catboost_model_training.py).
Its parameters are:
- lat_range: set to [41.25, 41.6], the latitude range used for extracting 
the input data
- lon_range: set to [1.9, 2.35], the longitude range used for the same thing
- ym_range: set to [YYYYMM, YYYYMM], with the starting and ending month 
(included) of the prediction
- n_harm: Nb of harmonics used for the Fourier series of the hour and month
input data, MUST MATCH WITH THE VALUE USED FOR TRAINING THE MODEL
- model_name and model_extension: name of the model used for the prediction

Other parameters include the NextCloud directory. Same as before, the 
user should set it up so that it corresponds to their Next Cloud directory.


## real_time_prediction.py
Script predicting high-resolution weather data on Barcelona from forecasting 
data from MeteoGalicia, using a trained weather downscaling model (catboost_model_training.py).
Its parameters are:
- lat_range: set to [41.25, 41.6], the latitude range used for extracting 
the input data
- lon_range: set to [1.9, 2.35], the longitude range used for the same thing
- nd: Nb of days preceding today where data should be predicted.
If equal to 1, only the forecasting data from today will be loaded and downscaled
Forecasting at one day means having the forecasting data for 96 hours from 
today midnight, so for 4 days. If set to 2, it will also do the same thing 
for the forecasting that was made yesterday, meaning forecasting data for 
96h starting yesterday midnight. This way, data overlaps for 3 days, but without 
being equal, as the forecasting made yesterday is different from the 
forecasting made today. And this goes on as 'nd' is increased.
- n_harm: Nb of harmonics used for the Fourier series of the hour and month
input data, MUST MATCH WITH THE VALUE USED FOR TRAINING THE MODEL
- model_name and model_extension: name of the model used for the prediction

Other parameters include the NextCloud directory. Same as before, the 
user should set it up so that it corresponds to their Next Cloud directory.

## utils.py
This file simply stores all the python functions used in the other scripts.

Copyright (c) 2024 Anceline, Arnau Comas, Jose Manuel Broto, Gerard Mor
