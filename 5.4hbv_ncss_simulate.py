import numpy as np
import pandas as pd
from hbv_model import hbv #imported from local python script
import joblib

#read basin list
basin_list = pd.read_csv('data/basins.csv', dtype={'id': str})
#only keep mardi and chepe basins
# basin_list = basin_list[basin_list['name'].isin(['mardi', 'chepe'])].reset_index(drop=True)

for id, name in zip(basin_list['id'], basin_list['name']):
    #read input csv file
    df = pd.read_csv(f"ncss_data/hbv_input/{name}_input.csv")
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
    df.to_csv(f'output/flow/output_{name}.csv', index=False)

# Load the saved lgbm model and make complete predictions
def add_lags(df, cols, nlags=3):
    for c in cols:
        for lag in range(1, nlags + 1):
            df[f'{c}_lag_{lag}'] = df[c].shift(lag)
    return df

for id, name in zip(basin_list['id'], basin_list['name']):
    hbv_flow = pd.read_csv(f'output/flow/output_{name}.csv')
    hbv_flow['date'] = pd.to_datetime(hbv_flow['date'])
    hbv_flow['year'] = hbv_flow['date'].dt.year
    hbv_flow['month'] = hbv_flow['date'].dt.month
    # add lags for precip and avgtemp
    hbv_flow = add_lags(hbv_flow, ['precip', 'avgtemp'], nlags=3)
    hbv_flow = hbv_flow.dropna().reset_index(drop=True)
    features = [col for col in hbv_flow.columns if col not in ['date', 'obs_flow', 'residual', 'latitude']]
    X = hbv_flow[features].astype(float)
    model = joblib.load(f'output/saved_models/lgbm_residual_model_{id}.pkl')
    hbv_flow['residual_pred'] = model.predict(X)
    hbv_flow['qsim_corrected'] = hbv_flow['qsim'] + hbv_flow['residual_pred']
    # only keep relevant columns for output
    hbv_flow = hbv_flow[['date', 'precip', 'avgtemp', 'qsim_corrected']]
    hbv_flow.rename(columns={'qsim_corrected': 'qsim'}, inplace=True)
    hbv_flow = hbv_flow.round(3)
    hbv_flow.to_csv(f'output/pp_flow/output_{name}.csv', index=False)