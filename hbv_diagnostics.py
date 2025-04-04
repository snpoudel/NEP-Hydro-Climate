import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('colorblind')

def rmse(obs, sim):
    return np.sqrt(np.mean((obs - sim) ** 2))
def nse(obs, sim):
    denominator = np.sum((obs - (np.mean(obs)))**2)
    numerator = np.sum((obs - sim)**2)
    nse_value = 1 - (numerator/denominator)
    return nse_value
def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    kge_value = r - np.sqrt((alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge_value

#basin list
basin_list = pd.read_csv('data/basins_og.csv', dtype={'id': str})

df_train = pd.DataFrame(columns=['id', 'nse', 'kge', 'rmse'])
df_test = pd.DataFrame(columns=['id', 'nse', 'kge', 'rmse'])

for id in basin_list['id']:
    #read output
    out_df = pd.read_csv(f'output/flow/output{id}.csv')
    out_df['date'] = pd.to_datetime(out_df['date'])
    out_df['year'] = out_df['date'].dt.year

    #train period
    train_start_year = basin_list[basin_list['id'] == id]['start_train_date'].values[0]
    train_end_year = basin_list[basin_list['id'] == id]['end_train_date'].values[0]

    #calculate nse, kge, rmse
    out_df_train = out_df[(out_df['year'] >= train_start_year) & (out_df['year'] <= train_end_year)].reset_index(drop=True)
    nse_train = nse(out_df_train['qobs'], out_df_train['qsim'])
    kge_train = kge(out_df_train['qobs'], out_df_train['qsim'])
    rmse_train = rmse(out_df_train['qobs'], out_df_train['qsim'])
    
    #test period
    test_start_year = basin_list[basin_list['id'] == id]['end_train_date'].values[0]
    test_end_year = basin_list[basin_list['id'] == id]['end_valid_date'].values[0]

    #calculate nse, kge, rmse
    out_df_test = out_df[(out_df['year'] > test_start_year) & (out_df['year'] <= test_end_year)].reset_index(drop=True)
    nse_test = nse(out_df_test['qobs'], out_df_test['qsim'])
    kge_test = kge(out_df_test['qobs'], out_df_test['qsim'])
    rmse_test = rmse(out_df_test['qobs'], out_df_test['qsim'])

    #append to dataframe
    df_train = pd.concat([df_train, pd.DataFrame([{'id': id, 'nse': nse_train, 'kge': kge_train, 'rmse': rmse_train}])], ignore_index=True)
    df_test = pd.concat([df_test, pd.DataFrame([{'id': id, 'nse': nse_test, 'kge': kge_test, 'rmse': rmse_test}])], ignore_index=True)

# Make a plot of NSE values for train and test period
plt.figure(figsize=(6, 4))
plt.plot(df_train['id'].astype(str).tolist(), df_train['nse'], marker='o', linestyle='-', label='Train', color='b')
plt.plot(df_test['id'].astype(str).tolist(), df_test['nse'], marker='s', linestyle='-', label='Test', color='r')
plt.xlabel('Basin ID')
plt.ylabel('NSE')
plt.ylim(0,None)
plt.title('NSE Values for Train and Test Periods')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
# plt.savefig('output/nse.png', dpi=300)
plt.show()
