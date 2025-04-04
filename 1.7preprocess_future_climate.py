import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

###########################################################################################################################################################################################################################################
#select basin to process
basin_list = pd.read_csv('data/basins.csv')
#read basin shapefile
basin_name = 'mardi'
basin_shapefile = gpd.read_file(f'data/gauges_shape_files/{basin_name}/watershed.shp')
#add a 0.25 degree buffer to the basin
basin_shapefile = basin_shapefile.buffer(0.1)
#calculate lower higher latitudes and longitudes
lat_min = basin_shapefile.bounds.miny[0]
lat_max = basin_shapefile.bounds.maxy[0]
lon_min = basin_shapefile.bounds.minx[0]
lon_max = basin_shapefile.bounds.maxx[0]

lat_range = np.arange(lat_min, lat_max+0.25, 0.25)
lon_range = np.arange(lon_min, lon_max+0.25, 0.25)

#make all combinations of lat and lon
lat_lon_combinations = np.array(np.meshgrid(lat_range, lon_range)).T.reshape(-1, 2)

###########################################################################################################################################################################################################################################
#read future climate data and calculate time series for selected basin
# model = ['CNRM-CM6-1', 'ACCESS-CM2']
model = ['ACCESS-CM2']
scenario = ['ssp245', 'ssp585']
product = ['pr', 'tasmax', 'tasmin']

for mod in model:
    for sc in scenario:
        for prod in product:
            #loop through years from 2020 to 2100
            df = pd.DataFrame() #to save time series data for each product
            for yr in range(2020, 2101):
                file_name = f'{prod}_day_{mod}_{sc}_r1i1p1f1_gn_{yr}.nc' #may need to change r1i1p1f1_gn depending on climate model
                #open file
                file = xr.open_dataset(f'ncss_data/{mod}/{sc}/{prod}/{file_name}')
                #go through all combinations of lat and lon and calculate average of a product
                list_of_data = []
                for comb in lat_lon_combinations:
                    lat = comb[0]
                    lon = comb[1]
                    #select data for the combination of lat and lon
                    temp_data = file[prod].sel(lat=comb[0], lon=comb[1], method='nearest')
                    #calculate total daily precip
                    temp_data = temp_data.resample(time='1D').sum()
                    #make array of precip values
                    data_array = temp_data.values
                    #add to list
                    list_of_data.append(data_array)
                #calculate average data from the list
                data_array = np.array(list_of_data)
                data_array = np.mean(data_array, axis=0)
                #convert to dataframe and add to df
                df_temp = pd.DataFrame(data_array, columns=[f'value'])
                #make date column and precip column
                df_temp['date'] = pd.date_range(start=f'{yr}-01-01', end=f'{yr}-12-31', freq='D')
                df_temp['date'] = pd.to_datetime(df_temp['date'])
                df = pd.concat([df, df_temp], axis=0)
                #save file
                df.to_csv(f'ncss_data/processed/{prod}_{mod}_{sc}.csv', index=False)
            print(f'Saved {prod}_{mod}_{sc}.csv')

