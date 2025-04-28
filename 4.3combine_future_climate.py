import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

## This script creates csv file containing precip and temp for each basin, model, and scenario

basins = ['mardi', 'chepe']
models = ['ACCESS-CM2', 'CNRM-CM6-1']
scenarios = ['ssp245', 'ssp585']

for basin in basins:
    for model in models:
        for scenario in scenarios:
            file_path = f'C:/Cornell/NEP-Hydro-Climate/ncss_data/processed/{basin}/{model}/{scenario}'
            os.chdir(file_path)

            # read precip, tempmax, tempmin
            precip = pd.read_csv(f'pr_{model}_{scenario}.csv')
            tempmax = pd.read_csv(f'tasmax_{model}_{scenario}.csv')
            tempmin = pd.read_csv(f'tasmin_{model}_{scenario}.csv')

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
            output_path = f'C:/Cornell/NEP-Hydro-Climate/ncss_data/processed/{basin}_{model}_{scenario}.csv'
            df.to_csv(output_path, index=False)
            print(f'Processed {basin} {model} {scenario}')