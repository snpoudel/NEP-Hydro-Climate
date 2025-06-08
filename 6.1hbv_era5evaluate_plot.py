import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('colorblind')
palette = sns.color_palette('colorblind')

# def rmse(obs, sim):
#     return np.sqrt(np.mean((obs - sim) ** 2))
def rmse(obs, sim):
    return np.mean(sim - obs)
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

df_train = pd.DataFrame(columns=['name', 'nse', 'kge', 'rmse'])
df_test = pd.DataFrame(columns=['name', 'nse', 'kge', 'rmse'])

for id in basin_list['name']:
    #read output
    id_to_true_id = {'mardi': 428, 'chepe': 440, 'khimti': 650, 'mai': 728}
    true_id = id_to_true_id.get(id)
    out_df = pd.read_csv(f'output/era5flow/output_{true_id}.csv')
    out_df['date'] = pd.to_datetime(out_df['date'])
    out_df['year'] = out_df['date'].dt.year

    #train period
    train_start_year = basin_list[basin_list['name'] == id]['start_train_date'].values[0]
    train_end_year = basin_list[basin_list['name'] == id]['end_train_date'].values[0]

    #calculate nse, kge, rmse
    out_df_train = out_df[(out_df['year'] >= train_start_year) & (out_df['year'] <= train_end_year)].reset_index(drop=True)
    nse_train = nse(out_df_train['Streamflow'], out_df_train['qsim'])
    kge_train = kge(out_df_train['Streamflow'], out_df_train['qsim'])
    rmse_train = rmse(out_df_train['Streamflow'], out_df_train['qsim'])
    
    #test period
    test_start_year = basin_list[basin_list['name'] == id]['end_train_date'].values[0]
    test_end_year = basin_list[basin_list['name'] == id]['end_valid_date'].values[0]

    #calculate nse, kge, rmse
    out_df_test = out_df[(out_df['year'] > test_start_year) & (out_df['year'] <= test_end_year)].reset_index(drop=True)
    nse_test = nse(out_df_test['Streamflow'], out_df_test['qsim'])
    kge_test = kge(out_df_test['Streamflow'], out_df_test['qsim'])
    rmse_test = rmse(out_df_test['Streamflow'], out_df_test['qsim'])

    #append to dataframe
    df_train = pd.concat([df_train, pd.DataFrame([{'name': id, 'nse': nse_train, 'kge': kge_train, 'rmse': rmse_train}])], ignore_index=True)
    df_test = pd.concat([df_test, pd.DataFrame([{'name': id, 'nse': nse_test, 'kge': kge_test, 'rmse': rmse_test}])], ignore_index=True)

#round values
df_train['nse'] = np.round(df_train['nse'], 2)
df_test['nse'] = np.round(df_test['nse'], 2)

basin_names = ['Mardi (ID: 428)', 'Chepe (ID: 440)', 'Khimti (ID: 650)', 'Mai (ID: 728)']
#replace names in dataframe
df_train['name'] = df_train['name'].replace({'mardi': basin_names[0], 'chepe': basin_names[1], 'khimti': basin_names[2], 'mai': basin_names[3]})
df_test['name'] = df_test['name'].replace({'mardi': basin_names[0], 'chepe': basin_names[1], 'khimti': basin_names[2], 'mai': basin_names[3]})

# #grouped bar plot for train and test nse across 4 basins
fig, ax = plt.subplots(figsize=(6, 4))
bar_width = 0.3
x = np.arange(len(df_train['name']))

# Plot bars
palette = sns.color_palette('colorblind')
ax.bar(x - bar_width/2, df_train['nse'], width=bar_width, label='Train', color=palette[0], edgecolor='black', hatch='//', alpha=0.8)
ax.bar(x + bar_width/2, df_test['nse'], width=bar_width, label='Test', color=palette[1], edgecolor='black', alpha=0.8)

# Add value labels on top of bars
for i in range(len(x)):
    # Train NSE
    ax.text(x[i] - bar_width/2, df_train['nse'][i] + 0.01,
            f"{df_train['nse'][i]:.2f}", ha='center', va='bottom', fontsize=8)
    # Test NSE
    ax.text(x[i] + bar_width/2, df_test['nse'][i] + 0.01,
            f"{df_test['nse'][i]:.2f}", ha='center', va='bottom', fontsize=8)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(df_train['name'])
ax.set_xlabel('Basin Name')
ax.set_ylabel('Nash-Sutcliffe Efficiency (NSE)')
ax.legend()
ax.set_ylim(0, 0.75)
ax.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('figures/nse_train_test.jpg', dpi=300)
plt.show()