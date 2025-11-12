# --- LGBM residual correction pipeline --------------------------------
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- helpers ---------------------------------------------------------------
def nse(obs, sim):
    obs, sim = np.asarray(obs), np.asarray(sim)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

def high_flow_bias(obs, sim, quantile=0.95):
    threshold = np.quantile(obs, quantile)
    index = obs >= threshold
    peaks_obs = obs[index]
    peaks_sim = sim[index]
    bias = np.mean(peaks_sim - peaks_obs) / np.mean(peaks_obs)
    return bias

def add_lags(df, cols, nlags=3):
    for c in cols:
        for lag in range(1, nlags + 1):
            df[f'{c}_lag_{lag}'] = df[c].shift(lag)
    return df

# --- load config / data ----------------------------------------------------
basins = pd.read_csv('data/basins_og.csv')
basin = basins.iloc[0]                # select a single basin
basin_id = basin['id']
basin_name = basin['name']
print(f"Processing basin: {basin_name} (ID: {basin_id})")
train_start, train_end = int(basin['start_train_date']), int(basin['end_train_date'])
valid_end = int(basin['end_valid_date'])

# df = pd.read_csv(f'output/flow/output_{basin_name}.csv', parse_dates=['date'])
df = pd.read_csv(f'output/era5flow/output_{basin_id}.csv', parse_dates=['date'])
# change col names temp to avgtemp and Streamflow to obs_flow for consistency
df.rename(columns={'temp': 'avgtemp', 'Streamflow': 'obs_flow'}, inplace=True)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month        # add seasonality
df['residual'] = df['obs_flow'] - df['qsim']

# --- feature engineering ---------------------------------------------------
df = add_lags(df, ['precip', 'avgtemp'], nlags=3)
df = df.dropna().reset_index(drop=True)

exclude = {'date', 'Date', 'obs_flow', 'residual', 'latitude'}
features = [c for c in df.columns if c not in exclude]

X = df[features].astype(float)
y = df['residual'].values

# --- train/validation split -------------------------------------------------
train_mask = (df['year'] >= train_start) & (df['year'] <= train_end)
valid_mask = (df['year'] > train_end) & (df['year'] <= valid_end)

X_train, y_train = X.loc[train_mask], y[train_mask]
X_val,   y_val   = X.loc[valid_mask], y[valid_mask]

# --- train LGBM model ------------------------------------------------------
# NSE metric for sklearn API
def nse_eval_metric(y_true, y_pred):
    obs = np.array(y_true)
    sim = np.array(y_pred)
    nse_val = 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    return 'NSE', nse_val, True  # True because higher NSE is better


model = LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    num_leaves=31,
    max_depth=7,
    learning_rate=0.05,
    n_estimators=20000,
    feature_fraction=0.7,
    bagging_fraction=0.6,
    bagging_freq=5,
    min_child_samples=20,
    min_split_gain=1e-3,
    lambda_l1=0.1,
    lambda_l2=0.5,
    random_state=42,
    verbosity=-1          # silences LightGBM logs
)

# Train model with NSE as eval metric
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=lambda y_pred, y_true: nse_eval_metric(y_true, y_pred),  # swap for sklearn API
    callbacks=[early_stopping(stopping_rounds=50)]
)

# save model to use later for prediction
joblib.dump(model, f'output/saved_models/lgbm_residual_model_{basin_id}.pkl')

# --- predict and compute metrics -------------------------------------------
df['pred_residual'] = model.predict(X, num_iteration=model.best_iteration_)
df['final_pred'] = df['qsim'] + df['pred_residual']

nse_train = nse(df.loc[train_mask, 'obs_flow'], df.loc[train_mask, 'final_pred'])
nse_valid = nse(df.loc[valid_mask, 'obs_flow'], df.loc[valid_mask, 'final_pred'])
peak_bias = high_flow_bias(df.loc[valid_mask, 'obs_flow'], df.loc[valid_mask, 'final_pred'])
nse_valid_before = nse(df.loc[valid_mask, 'obs_flow'], df.loc[valid_mask, 'qsim'])
peak_bias_before = high_flow_bias(df.loc[valid_mask, 'obs_flow'], df.loc[valid_mask, 'qsim'])

print(f'Train NSE: {nse_train:.4f}')
print(f'Validation NSE: {nse_valid:.4f}')
print(f'Validation Peak Flow Bias: {peak_bias:.4f}')
print(f'Validation NSE (Before Correction): {nse_valid_before:.4f}')
print(f'Validation Peak Flow Bias (Before Correction): {peak_bias_before:.4f}')

# --- save corrected output -------------------------------------------------
df = df[['date', 'precip', 'avgtemp', 'obs_flow', 'final_pred']]
df.rename(columns={'obs_flow': 'Streamflow', 'final_pred': 'qsim'}, inplace=True)
df.to_csv(f'output/era5_ppflow/output_{basin_id}.csv', index=False)