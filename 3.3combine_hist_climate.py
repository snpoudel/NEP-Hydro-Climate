import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

## This script creates csv file containing precip and temp for each basin, model, and scenario

#read basin list
basin_list = pd.read_csv('data/basins_og.csv')

basins = ['mardi', 'chepe', 'khimti', 'mai']
models = ['ACCESS-CM2', 'CNRM-CM6-1']
scenarios = ['historical']

for basin in basins:
    for model in models:
        for scenario in scenarios:
            file_path = f'ncss_data/processed/{basin}/{model}/{scenario}'
            # read precip, tempmax, tempmin
            precip = pd.read_csv(f'{file_path}/pr_{model}_{scenario}.csv')
            tempmax = pd.read_csv(f'{file_path}/tasmax_{model}_{scenario}.csv')
            tempmin = pd.read_csv(f'{file_path}/tasmin_{model}_{scenario}.csv')

            # convert precip to mm/day, temp to C
            precip['value'] = precip['value'] * 86400  # mm/day
            tempmax['value'] = tempmax['value'] - 273.15  # C
            tempmin['value'] = tempmin['value'] - 273.15  # C

            # make a dataframe with the date and the three variables
            df = pd.DataFrame()
            df['date'] = precip['date']
            df['precip'] = precip['value']
            df['avgtemp'] = (tempmax['value'] + tempmin['value']) / 2

            # save the dataframe to a csv file
            if not os.path.exists('ncss_data/processed'):
                os.makedirs('ncss_data/processed')
            output_path = f'ncss_data/processed/{basin}_{model}_{scenario}.csv'
            df.to_csv(output_path, index=False)
            print(f'Processed {basin} {model} {scenario}')


#add observed streamflow data and prepare final input file for model
for basin in basins:
    #aggregate climate models
    access = pd.read_csv(f'ncss_data/processed/{basin}_ACCESS-CM2_historical.csv')
    cnrm = pd.read_csv(f'ncss_data/processed/{basin}_CNRM-CM6-1_historical.csv')
    #combine the two models
    df = pd.DataFrame()
    df['date'] = access['date']
    df['date'] = pd.to_datetime(df['date'])
    df['precip'] = (access['precip'] + cnrm['precip']) / 2
    df['avgtemp'] = (access['avgtemp'] + cnrm['avgtemp']) / 2

    #add latitude to df
    lat = basin_list[basin_list['name'] == basin]['latitude'].values[0]
    df['latitude'] = lat

    #read observed streamflow data
    obs = pd.read_csv(f'data/streamflow/{basin}_streamflow.csv')
    obs['Date'] = pd.to_datetime(obs['Date'])
    obs =obs.rename(columns={'Date':'date', 'Streamflow':'obs_flow'})

    #add streamflow to df and merge by date
    df = pd.merge(df, obs, on='date', how='left')

    #convert streamflow from m3/s to mm/day
    basin_area = basin_list[basin_list['name'] == basin]['area_km2'].values[0]  # km2
    df['obs_flow'] = df['obs_flow'] * 86400 / (basin_area * 1000)  # mm/day

    #save the final input file
    df.to_csv(f'ncss_data/hbv_input/{basin}_input.csv', index=False)


