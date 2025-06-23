import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import genextreme
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sns.set_palette('colorblind')
palette = sns.color_palette('colorblind')
colors = {
    'SSP245': palette[1],
    'SSP585': palette[2]
}

# Parametric bootstrapping from fitted GEV
def bootstrap_gev_return_levels(data, return_periods, n_bootstrap=10000):
    """
    Perform parametric bootstrapping for GEV return levels.
    - Fit GEV to original data
    - Generate synthetic samples from fitted distribution
    - Refit GEV to each sample and compute return levels
    """
    shape, loc, scale = genextreme.fit(data)
    q_boot = np.zeros((n_bootstrap, len(return_periods)))

    for i in range(n_bootstrap):
        synthetic_sample = genextreme.rvs(shape, loc=loc, scale=scale, size=len(data))
        try:
            shape_b, loc_b, scale_b = genextreme.fit(synthetic_sample)
            q_boot[i, :] = genextreme.ppf(1 - 1/return_periods, shape_b, loc=loc_b, scale=scale_b)
        except Exception:
            q_boot[i, :] = np.nan

    return q_boot

# Return periods and labels
return_periods = np.array([25, 50, 100])
rp_labels = ['25-year', '50-year', '100-year']

# Load basin info
basin_list = pd.read_csv('data/basins.csv', dtype={'id': str})
basin_list = basin_list[basin_list['name'].isin(['chepe', 'mardi'])]

# Precompute everything once
results = {}

for i, row in basin_list.iterrows():
    basin_id = row['id']
    basin_name = row['name']

    # Load data
    hist = pd.read_csv(f'output/flow/output_{basin_name}.csv')
    ssp245 = pd.read_csv(f'output/future_flow/output_{basin_name}_ssp245.csv')
    ssp585 = pd.read_csv(f'output/future_flow/output_{basin_name}_ssp585.csv')

    for df in [hist, ssp245, ssp585]:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

    # Annual maxima
    hist_max = hist.groupby('year')['qsim'].max().dropna()
    ssp245_max = ssp245.groupby('year')['qsim'].max().dropna()
    ssp585_max = ssp585.groupby('year')['qsim'].max().dropna()

    # Bootstrap return levels
    hist_q = bootstrap_gev_return_levels(hist_max.values, return_periods)
    q245 = bootstrap_gev_return_levels(ssp245_max.values, return_periods)
    q585 = bootstrap_gev_return_levels(ssp585_max.values, return_periods)

    # Store relative change
    results[basin_name] = {
        'hist': hist_q,
        'ssp245': ((q245 - hist_q) / hist_q) * 100,
        'ssp585': ((q585 - hist_q) / hist_q) * 100
    }

# Plot: one figure per basin (3-row, 1-col for 25/50/100 year changes)
for basin_name in ['chepe', 'mardi']:
    basin_id = basin_list[basin_list['name'] == basin_name]['id'].values[0]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 8), sharex=False)

    for rp_idx, rp_label in enumerate(rp_labels):
        ax = axes[rp_idx]

        delta_245 = results[basin_name]['ssp245'][:, rp_idx]
        delta_585 = results[basin_name]['ssp585'][:, rp_idx]

        # Remove inf/nan
        delta_245 = delta_245[np.isfinite(delta_245)]
        delta_585 = delta_585[np.isfinite(delta_585)]

        # Plot histograms
        sns.histplot(delta_245, bins=30, kde=True, color=colors['SSP245'],
                     label='SSP245', ax=ax, alpha=0.5)
        sns.histplot(delta_585, bins=30, kde=True, color=colors['SSP585'],
                     label='SSP585', ax=ax, alpha=0.5)

        # Medians
        med_245 = np.nanmedian(delta_245)
        med_585 = np.nanmedian(delta_585)
        ax.axvline(med_245, color=colors['SSP245'], linestyle='--', label='Median SSP245')
        ax.axvline(med_585, color=colors['SSP585'], linestyle='--', label='Median SSP585')

        # Basic bootstrap CIs
        ci_245_lo = 2 * med_245 - np.nanpercentile(delta_245, 97.5)
        ci_245_hi = 2 * med_245 - np.nanpercentile(delta_245, 2.5)
        ci_585_lo = 2 * med_585 - np.nanpercentile(delta_585, 97.5)
        ci_585_hi = 2 * med_585 - np.nanpercentile(delta_585, 2.5)

        # Add CI text box
        ci_text = (f"SSP245: {med_245:.1f}% [{ci_245_lo:.1f}, {ci_245_hi:.1f}]\n"
                   f"SSP585: {med_585:.1f}% [{ci_585_lo:.1f}, {ci_585_hi:.1f}]")
        ax.text(
            0.99, 0.97,
            ci_text,
            transform=ax.transAxes,
            fontsize=8,
            ha='right',
            va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8)
        )

        ax.set_xlabel(f'Percentage (%) Change in {rp_label} Flood Relative to Historical')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.5)
        if rp_idx == 0:
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle(f"Relative Change in Extreme Flood Quantiles for {basin_name.capitalize()} Basin (ID: {basin_id})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.savefig(f'figures/relative_change_by_rp_{basin_name}.jpg', dpi=300)
    plt.close()
