import pandas as pd
import numpy as np

#read lat lon data csv file
latlon = pd.read_csv('data/filtered_hpp_with_gauges.csv')

#convert lat lon which is in degrees minutes seconds seperated by space to decimal
latlon['lat_decimal'] = latlon['lat'].apply(lambda x: float(x.split()[0]) + float(x.split()[1])/60 + float(x.split()[2])/3600)
latlon['lon_decimal'] = latlon['lon'].apply(lambda x: float(x.split()[0]) + float(x.split()[1])/60 + float(x.split()[2])/3600)

#use to three decimal places
latlon['lat_decimal'] = latlon['lat_decimal'].apply(lambda x: round(x, 3))
latlon['lon_decimal'] = latlon['lon_decimal'].apply(lambda x: round(x, 3))

#save to csv
latlon.to_csv('data/filtered_hpp_with_gauges_decimal.csv', index=False)


