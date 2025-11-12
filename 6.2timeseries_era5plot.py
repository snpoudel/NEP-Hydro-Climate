import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('colorblind')
palette = sns.color_palette('colorblind')

# Load basin metadata
basin_list = pd.read_csv('data/basins_og.csv', dtype={'id': str})

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.5)

for i, id in enumerate(basin_list['id']):
    out_df = pd.read_csv(f'output/era5_ppflow/output_{id}.csv')
    out_df['date'] = pd.to_datetime(out_df['date'])

    # Get test period end
    test_end_year = basin_list.loc[basin_list['id'] == id, 'end_valid_date'].values[0]
    test_end_date = pd.to_datetime(f'{test_end_year}-12-31')

    # Filter last 365 days
    out_df = out_df[out_df['date'] <= test_end_date]
    out_df_test = out_df.tail(365).reset_index(drop=True)

    # Main plot: streamflow
    ax1 = axes[i]
    ax1.plot(out_df_test['date'], out_df_test['Streamflow'], label='Observed', color=palette[0])
    ax1.plot(out_df_test['date'], out_df_test['qsim'], label='Simulated', color=palette[1], marker='^', markersize=4, markevery=5)
    ax1.set_ylabel('Streamflow (mm/day)')
    basin_name = basin_list.loc[basin_list['id'] == id, 'name'].values[0].capitalize()
    ax1.set_title(f'{basin_name} (ID: {id})')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Secondary axis: precipitation bars (inverted)
    ax2 = ax1.twinx()
    ax2.bar(out_df_test['date'], out_df_test['precip'], width=1.0, color='gray', alpha=0.5)
    ax2.invert_yaxis()  # Precip appears to fall from the top
    ax2.set_ylabel('Precipitation (mm/day)', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray',)

    if i == 3:
        ax1.set_xlabel('Date')
    if i == 0:
        ax1.legend(loc='best')

plt.tight_layout()
plt.savefig('figures/timeseries_plot.jpg', dpi=300)
plt.savefig('figures/inkscape/timeseries_plot.svg')
# plt.close()
