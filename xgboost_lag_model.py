import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

station_id = pd.read_csv('data/basins.csv', dtype={'id': str})



station_id = '728'
start_train = 1983
end_train = 1996
end_test = 2001
df = pd.read_csv(f'output/flow/output{station_id}.csv')

#only keep date after start_train
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year >= start_train].reset_index(drop=True)
#only keep date before end_test
df = df[df['date'].dt.year <= end_test].reset_index(drop=True)

#prepare data for xgboost format
df_boost = df.copy()
df_boost['year'] = df_boost['date'].dt.year
df_boost['month'] = df_boost['date'].dt.month
df_boost['day'] = df_boost['date'].dt.day
df_boost['residual'] = df_boost['qobs'] - df_boost['qsim']
df_boost.drop(columns=['qobs', 'qsim', 'date', 'latitude'], inplace=True)

#add lagged features
# Add lagged features
for lag in range(1, 4): #add lagged features for 1 to 3 days
    for col in ['temp', 'precip', 'residual']:
        df_boost[f'{col}_lag{lag}'] = df_boost[col].shift(lag)
df_boost = df_boost.dropna()

#use 1994 to 2009 for training and 2009 to 2015 for testing
train = df_boost[df_boost['year'] <= end_train].reset_index(drop=True)
test = df_boost[df_boost['year'] > end_train].reset_index(drop=True)

#prepare xtrain and ytrain for training
X_train = train.drop(columns=['residual'])
y_train = train['residual']

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [2, 5, 10], 
    'learning_rate': [0.01, 0.1, 0.3], 
    # 'subsample': [0.5, 0.7, 1],
    # 'colsample_bytree': [0.5, 0.7, 1],
}

# Initialize the XGBRegressor
xgb = XGBRegressor(random_state=0)

print("XGBoost Hyperparameters tuning...")
# Perform Grid Search to find the best parameters
xgb_grid = GridSearchCV(xgb, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

# Get the best model
best_xgb = xgb_grid.best_estimator_

# Display the best hyperparameters and cross-validation mean squared error, root mean squared error, and r2 score
print(f"Best XGBoost Parameters:\n{xgb_grid.best_params_}")
# print(f"CV MSE = {-xgb_grid.best_score_:.3f}")
print(f"CV RMSE = {np.sqrt(-xgb_grid.best_score_):.2f}")

#fit the best model on the entire training dataset
print('Fitting best XGBoost on entire training dataset...')
best_xgb.fit(X_train, y_train)

#evaluate the model on the test set, rmse and r2 score
X_test = test.drop(columns=['residual'])
y_test = test['residual']
y_pred = best_xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.2f} & R2 Score: {r2:.2f}")
