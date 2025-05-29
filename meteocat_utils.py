import requests
import polars as pl
from io import StringIO
import os
import numpy as np

def get_page_socrata_dataset(url, limit, df=None, parquet_file=None):

    r = requests.get(url)
    while r.status_code != 200:
        r = requests.get(url)
    r.encoding = 'utf-8'
    # sample_df = pl.read_csv(StringIO(r.text), n_rows=1)
    # schema = {col: pl.Utf8 for col in sample_df.columns}
    df_ = pl.read_csv(StringIO(r.text), infer_schema=False)
    if len(df_) < limit:
        next_ = False
    else:
        next_ = True
    if len(df_)>0:
        if df is not None and len(df)>0:
            # Ensure both DataFrames have the same schema
            for col in df.columns:
                if col not in df_.columns:
                    df_ = df_.with_column(pl.lit(None).alias(col).cast(pl.Utf8))
            for col in df_.columns:
                if col not in df.columns:
                    df = df.with_column(pl.lit(None).alias(col).cast(pl.Utf8))
            df_ = df.vstack(df_)
    else:
        df_ = df
    if parquet_file is not None:
        df_.write_parquet(f"{parquet_file}.parquet")
    return {'df': df_, 'continue': next_}

def get_filenames_socrata_dataset(file_dir, file_basename, file_format):
    files = os.listdir(file_dir)
    files = [f for f in files if f"{file_basename}_" in f and f.endswith(file_format)]
    files = [f"{file_dir}/{f}" for f in files]
    return files

def get_complete_socrata_dataset(url, parquet_file, limit, update=True, one_file=True):

    offset = 0
    url = f"{url}$select=:*, *&$order=:updated_at"
    i=0
    last_parquet_file = parquet_file
    if not one_file:
        files = get_filenames_socrata_dataset(os.path.dirname(parquet_file), os.path.basename(parquet_file),
                                              file_format=".parquet")
        if len(files)>0:
            i = max([int(f.replace(".parquet","").split("_")[-1]) for f in files])
            last_parquet_file = f"{parquet_file}_{i}"
            i = i + 1
    if os.path.exists(f"{last_parquet_file}.parquet"):
        print(f"Reusing the already downloaded data, loading the dataset...")
        df = pl.read_parquet(f"{last_parquet_file}.parquet")
        last_date = df.select(pl.col(':updated_at').max()).to_numpy()[0][0]
        print(f"last_date:{last_date}")
        url = f"{url}&$where=:updated_at > '{last_date}'"
    else:
        df = None

    if update or df is None:
        print(f"Downloading dataset from {offset + 1} to {offset + limit}")
        result = get_page_socrata_dataset(f"{url}&$limit={limit}&$offset={offset}", limit,
                                          df=df if one_file else None,
                                          parquet_file= None if one_file else f"{parquet_file}_{i}")
        del df
        while result['continue']:
            i = i + 1
            offset += limit
            print(f"\tfrom {offset + 1} to {offset + limit}")
            result = get_page_socrata_dataset(f"{url}&$limit={limit}&$offset={offset}", limit,
                                              df=result['df'] if one_file else None,
                                              parquet_file= None if one_file else f"{parquet_file}_{i}")
        if one_file:
            result['df'] = result['df'].unique(subset=[':id'])
            result['df'].write_parquet(f"{parquet_file}.parquet")
            return result['df']
        else:
            print(f"All datasets were stored in {os.path.dirname(parquet_file)}")
    else:
        return df
