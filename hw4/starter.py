#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys


categorical = ['PUlocationID', 'DOlocationID']


def read_data(filename):

    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 2
    file_name = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = 'output_' + file_name.split('/')[-1]

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(file_name)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    # mean prediction
    print(y_pred.mean())

    df_result = pd.DataFrame({'pred': y_pred})

    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df_result.index.astype('str')

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == '__main__':
    run()





