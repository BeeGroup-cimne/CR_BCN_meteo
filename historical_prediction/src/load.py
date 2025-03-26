import hashlib
import json
import os
import numpy as np
from beelib.beehbase import save_to_hbase
# from utils.hbase import save_to_hbase
from neo4j import GraphDatabase
import pandas as pd
import beelib
# import settings


def harmonize_endesa(df_original):
    config = beelib.beeconfig.read_config('historical_prediction/config.json')
    morph_config = 'historical_prediction/src/mapping.yaml'
    # df_final_bcn = pd.read_parquet(
    #     '/Users/jose/Nextcloud/Beegroup/Projects/ClimateReady-BCN/WP3-VulnerabilityMap/Weather Downscaling/Models_and_predictions/Historical_ERA5Land/Predictions/prediction_202006-202007.parquet')
    df_original["weatherId"] = df_original['weatherStation'].apply(lambda x: (x + '-weatherera').encode("utf-8"))
    df_original["weatherId"] = df_original['weatherId'].apply(lambda x: hashlib.sha256(x).hexdigest())
    df_original[['latitude', 'longitude']] = df_original['weatherStation'].str.split('_', expand=True).astype(float)
    # df_original['time'] = pd.to_datetime(df_original['time'], errors='coerce')

    # Load to Neo4j
    df = df_original.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.drop_duplicates(subset=['weatherStation'], inplace=True, keep='first')
    documents = {"weather": df.to_dict(orient='records')}
    beelib.beetransformation.map_and_save(documents, morph_config, config)
    # beelib.beetransformation.map_and_print(documents, morph_config, config)
    # Load to HBase
    freq = "P1H"
    user = "CRBCN"
    ts_buckets = 10000000
    buckets = 4

    df_original['relativeHumidity'] = df_original['relativeHumidity'] * 100
    df_original['start'] = df_original['time'].astype(int) // 10 ** 9
    df_original['end'] = (df_original['time'] + pd.Timedelta(hours=1)).astype(int) // 10 ** 9
    df_original["bucket"] = (df_original['start'].apply(int) // ts_buckets) % buckets
    df_original['isReal'] = False
    # df_original.to_csv("historical_prediction/data/predictions/era_downscalling_202206-202209.csv")
    hbase_conn = config['hbase_store_harmonized_data']
    weather_device_table = f"harmonized_online_DryBulbTemperatureDownScalingEra5Land_100_SUM_{freq}_{user}"
    save_to_hbase(df_original.to_dict(orient="records"), weather_device_table, hbase_conn,
                  [("info", ['end', 'isReal']), ("v", ['airTemperature'])],
                  row_fields=['bucket', 'weatherId', 'start'])

    weather_device_table = f"harmonized_online_RelativeHumidityDownScalingEra5Land_100_SUM_{freq}_{user}"
    save_to_hbase(df_original.to_dict(orient="records"), weather_device_table, hbase_conn,
                  [("info", ['end', 'isReal']), ("v", ['relativeHumidity'])],
                  row_fields=['bucket', 'weatherId', 'start'])

    # weather_period_table = f"harmonized_batch_DryBulbTemperatureDownScaling_100_SUM_{freq}_{user}"
    # save_to_hbase(df_final_bcn.to_dict(orient="records"), weather_period_table, hbase_conn,
    #               [("info", ['end', 'isReal']), ("v", ['airTemperature'])],
    #               row_fields=['bucket', 'start', 'weatherId'])
    #
    # weather_period_table = f"harmonized_batch_RelativeHumidityDownScaling_100_SUM_{freq}_{user}"
    # save_to_hbase(df_final_bcn.to_dict(orient="records"), weather_period_table, hbase_conn,
    #               [("info", ['end', 'isReal']), ("v", ['relativeHumidity'])],
    #               row_fields=['bucket', 'start', 'weatherId'])
