import numpy as np
import pandas as pd
from hbv_model import hbv #imported from local python script

#read basin list
basin_list = pd.read_csv('data/basins.csv', dtype={'id': str})

for id in basin_list['id']:
    #read precipitation data
    precip_df = pd.read_csv(f'data/era5_precip/precip_{id}.csv')
    #read temperature data
    temp_df = pd.read_csv(f'data/era5_temp/temp_{id}.csv')
    temp_df['temp'] = temp_df['temp'] - 273.15 #convert kelvin to celsius
    #extract latitude
    lat = basin_list[basin_list['id'] == id]['latitude'].values[0]

    #make dataframe with date, precip, temp, latitude
    df = pd.DataFrame({'date': precip_df['date'], 'precip': precip_df['precip'], 'temp': temp_df['temp'], 'latitude': lat})

    #read calibrated parameters
    hbv_params = pd.read_csv(f'output/parameter/param_{id}.csv')
    hbv_params.drop(columns=['station_id'], inplace=True)
    hbv_params = hbv_params.values.flatten()

    #run hbv model
    qsim = hbv(hbv_params, df['precip'], df['temp'], df['date'], df['latitude'], routing=1)

    #add qsim to dataframe
    df['qsim'] = qsim

    #read observed flow data
    obs_df = pd.read_csv(f'data/streamflow/{id}streamflow.csv')
    #convert flowfrom m3/s to mm/day
    basin_area = basin_list[basin_list['id'] == id]['area_km2'].values[0]
    obs_df['Streamflow'] = obs_df['Streamflow'] * 86400 / (basin_area * 1000)
    #change column name
    obs_df = obs_df.rename(columns={'Date': 'date', 'Streamflow': 'qobs'})
    #change any negative values to nan
    obs_df.loc[obs_df['qobs'] < 0, 'qobs'] = np.nan

    #merge obs_df and df by date
    df = pd.merge(df, obs_df, on='date', how='left')

    #save output
    df.to_csv(f'output/flow/output{id}.csv', index=False)