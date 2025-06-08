import numpy as np
import pandas as pd
from hbv_model import hbv #imported from local python script

#read basin list
basin_list = pd.read_csv('data/basins.csv', dtype={'id': str})
#only use chepe and mardi basins as they showed good evaluation performance
basin_list = basin_list[basin_list['name'].isin(['chepe', 'mardi'])]

scenario = 'ssp585'

#read basin list
basin_list = pd.read_csv('data/basins.csv', dtype={'id': str})
#only keep mardi and chepe basins
basin_list = basin_list[basin_list['name'].isin(['mardi', 'chepe'])].reset_index(drop=True)

for id in basin_list['name']:
    #read  data
    access = pd.read_csv(f'ncss_data/processed/{id}_ACCESS-CM2_{scenario}.csv')
    cnrm = pd.read_csv(f'ncss_data/processed/{id}_CNRM-CM6-1_{scenario}.csv')
    #calculate average of two models for precip and avgtemp columns
    df = pd.DataFrame({'date': access['date'], 'precip': (access['precip'] + cnrm['precip']) / 2, 'temp': (access['avgtemp'] + cnrm['avgtemp']) / 2})
    lat = basin_list[basin_list['name'] == id]['latitude'].values[0]
    #add latitude to df
    df['latitude'] = lat

    #read calibrated parameters
    if id == 'mardi':
        true_id = 428
    elif id == 'chepe':
        true_id = 440
    hbv_params = pd.read_csv(f'output/parameter/param_{true_id}.csv')
    hbv_params.drop(columns=['station_id'], inplace=True)
    hbv_params = hbv_params.values.flatten()

    #run hbv model
    qsim = hbv(hbv_params, df['precip'], df['temp'], df['date'], df['latitude'], routing=1)

    #add qsim to dataframe
    df['qsim'] = qsim

    #round to 3 decimal places for all columns
    df = df.round(3)

    #save output
    df.to_csv(f'output/future_flow/output_{id}_{scenario}.csv', index=False)