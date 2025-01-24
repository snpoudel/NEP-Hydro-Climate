import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

########################################################################################################################################################################################################################
##--Precip--##
#read filtered basin with at least 2 gauges
basin_list = pd.read_csv('data/streamflow/basins.csv')
for iter in range(len(basin_list)):
    basin_id = basin_list['id'][iter]
    basin_name = basin_list['name'][iter]

    #read basin shapefile
    basin_shapefile = gpd.read_file(f'data/gauges_shape_files/{basin_name}/watershed.shp')
    #add a 0.25 degree buffer to the basin
    basin_shapefile = basin_shapefile.buffer(0.1)
    #plot basin
    # basin_shapefile.plot()
    #calculate lower higher latitudes and longitudes
    lat_min = basin_shapefile.bounds.miny[0]
    lat_max = basin_shapefile.bounds.maxy[0]
    lon_min = basin_shapefile.bounds.minx[0]
    lon_max = basin_shapefile.bounds.maxx[0]

    df_all = pd.DataFrame()
    for yr in range(1940,2025):
        #read era5 precipitation nc file
        era5_precip = xr.open_dataset(f'data/era5_new/era5_{yr}/data_stream-oper_stepType-accum.nc')

        #find latitude and longitude closest to the basin
        lat_min = era5_precip.sel(latitude=lat_min, method='nearest').latitude.values
        lat_max = era5_precip.sel(latitude=lat_max, method='nearest').latitude.values
        lon_min = era5_precip.sel(longitude=lon_min, method='nearest').longitude.values
        lon_max = era5_precip.sel(longitude=lon_max, method='nearest').longitude.values

        lat_range = np.arange(lat_min, lat_max+0.25, 0.25)
        lon_range = np.arange(lon_min, lon_max+0.25, 0.25)

        #make all combinations of lat and lon
        lat_lon_combinations = np.array(np.meshgrid(lat_range, lon_range)).T.reshape(-1, 2)

        list_of_precip = []
        for comb in lat_lon_combinations:
            lat = comb[0]
            lon = comb[1]
            temp_precip = era5_precip.tp.sel(latitude=comb[0], longitude=comb[1])
            #calculate total daily precip
            temp_precip = temp_precip.resample(valid_time='1D').sum()
            #make array of precip values
            precip_array = temp_precip.values
            #convert to mm
            precip_array = precip_array * 1000
            #add to list
            list_of_precip.append(precip_array)

        #calculate average precip from the list
        average_precip = np.mean(list_of_precip, axis=0)
        #make dataframe with date and precip values
        date = era5_precip.valid_time.values
        #only keep date without time in the date
        date = [str(i).split('T')[0] for i in date]
        date = pd.to_datetime(date)
        #only keep unique dates
        date = np.unique(date)
        df = pd.DataFrame({'date': date, 'precip': average_precip})
        #append to the main dataframe
        df_all = pd.concat([df_all, df]).reset_index(drop=True)
    #save the dataframe
    df_all.to_csv(f'data/era5_precip/precip_{basin_id}.csv', index=False)



########################################################################################################################################################################################################################
##--Temp--##
#read filtered basin with at least 2 gauges
basin_list = pd.read_csv('data/streamflow/basins.csv')
for iter in range(len(basin_list)):
    basin_id = basin_list['id'][iter]
    basin_name = basin_list['name'][iter]

    #read basin shapefile
    basin_shapefile = gpd.read_file(f'data/gauges_shape_files/{basin_name}/watershed.shp')
    #add a 0.25 degree buffer to the basin
    basin_shapefile = basin_shapefile.buffer(0.1)
    #plot basin
    # basin_shapefile.plot()
    #calculate lower higher latitudes and longitudes
    lat_min = basin_shapefile.bounds.miny[0]
    lat_max = basin_shapefile.bounds.maxy[0]
    lon_min = basin_shapefile.bounds.minx[0]
    lon_max = basin_shapefile.bounds.maxx[0]

    df_all = pd.DataFrame()
    for yr in range(1940,2025):
        #read era5 precipitation nc file
        era5_temp = xr.open_dataset(f'data/era5_new/era5_{yr}/data_stream-oper_stepType-instant.nc')

        #find latitude and longitude closest to the basin
        lat_min = era5_temp.sel(latitude=lat_min, method='nearest').latitude.values
        lat_max = era5_temp.sel(latitude=lat_max, method='nearest').latitude.values
        lon_min = era5_temp.sel(longitude=lon_min, method='nearest').longitude.values
        lon_max = era5_temp.sel(longitude=lon_max, method='nearest').longitude.values

        lat_range = np.arange(lat_min, lat_max+0.25, 0.25)
        lon_range = np.arange(lon_min, lon_max+0.25, 0.25)

        #make all combinations of lat and lon
        lat_lon_combinations = np.array(np.meshgrid(lat_range, lon_range)).T.reshape(-1, 2)

        list_of_precip = []
        for comb in lat_lon_combinations:
            lat = comb[0]
            lon = comb[1]
            temp_precip = era5_temp.t2m.sel(latitude=comb[0], longitude=comb[1])
            #calculate average daily temp
            temp_precip = temp_precip.resample(valid_time='1D').mean()
            #make array of precip values
            precip_array = temp_precip.values
            #add to list
            list_of_precip.append(precip_array)

        #calculate average precip from the list
        average_precip = np.mean(list_of_precip, axis=0)
        #make dataframe with date and precip values
        date = era5_temp.valid_time.values
        #only keep date without time in the date
        date = [str(i).split('T')[0] for i in date]
        date = pd.to_datetime(date)
        #only keep unique dates
        date = np.unique(date)
        df = pd.DataFrame({'date': date, 'temp': average_precip})
        #append to the main dataframe
        df_all = pd.concat([df_all, df]).reset_index(drop=True)
    #save the dataframe
    df_all.to_csv(f'data/era5_temp/temp_{basin_id}.csv', index=False)