import numpy as np
import pandas as pd

#read basin list
basin_list = pd.read_csv('data/basins.csv')

for id, lat in zip(basin_list['id'], basin_list['latitude']):
    # Read and process streamflow data
    streamflow = pd.read_csv(f'data/streamflow/{id}streamflow.csv')
    #convert -9999 to NaN
    streamflow.replace(-999, np.nan, inplace=True)
    streamflow['date'] = pd.to_datetime(streamflow['Date'])
    
    # Read and process precipitation data
    precip = pd.read_csv(f'data/era5_precip/precip_{id}.csv')
    precip['date'] = pd.to_datetime(precip['date'])
    
    # Read and process temperature data
    temp = pd.read_csv(f'data/era5_temp/temp_{id}.csv')
    temp['date'] = pd.to_datetime(temp['date'])
    temp['temp'] = temp['temp'] - 273.15  # Convert temperature to Celsius
    
    # Merge dataframes on 'date'
    combined = streamflow.merge(precip, on='date').merge(temp, on='date')

    # Add Latitude to the combined data
    combined['latitude'] = lat
    combined['year'] = combined['date'].dt.year

    #convert streamflow from m3/s to mm/day
    basin_area = basin_list[basin_list['id'] == id]['area_km2'].values[0]
    combined['Streamflow'] = combined['Streamflow'] * 86400 / (basin_area * 1000)

    # Save the combined data
    combined.to_csv(f'data/processed_data/input{id}.csv', index=False)