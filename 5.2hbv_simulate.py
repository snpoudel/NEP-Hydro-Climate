import numpy as np
import pandas as pd
from hbv_model import hbv #imported from local python script

#read basin list
basin_list = pd.read_csv('data/basins.csv', dtype={'id': str})
#only keep mardi and chepe basins
basin_list = basin_list[basin_list['name'].isin(['mardi', 'chepe'])].reset_index(drop=True)

for id in basin_list['name']:
    #read input csv file
    df = pd.read_csv(f"ncss_data/hbv_input/{id}_input.csv")
    #read calibrated parameters
    hbv_params = pd.read_csv(f'output/parameter/param_{id}.csv')
    hbv_params.drop(columns=['station_id'], inplace=True)
    hbv_params = hbv_params.values.flatten()

    #run hbv model
    qsim = hbv(hbv_params, df['precip'], df['avgtemp'], df['date'], df['latitude'], routing=1)

    #add qsim to dataframe
    df['qsim'] = qsim

    #round to 3 decimal places for all columns
    df = df.round(3)

    #save output
    df.to_csv(f'output/flow/output_{id}.csv', index=False)