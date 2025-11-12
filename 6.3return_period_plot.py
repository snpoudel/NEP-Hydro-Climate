import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import genextreme
from tqdm import tqdm

sns.set_palette('colorblind')
palette = sns.color_palette('colorblind')
colors = {
    'Historical': palette[0],  # blue
    'SSP245': palette[1],      # orange
    'SSP585': palette[2]       # red
}

# Function to extract annual maxima from a DataFrame
def get_annual_maxima(df):
    df['year'] = df['date'].dt.year
    return df.groupby('year')['qsim'].max().dropna()

# Function to compute empirical return levels
def empirical_return_levels(data):
    sorted_data = np.sort(data)[::-1]  # descending
    n = len(sorted_data)
    ranks = np.arange(1, n+1)
    rp = (n + 1) / ranks  # Weibull plotting position
    return rp, sorted_data


def fit_gev_and_ci(data, return_periods, n_bootstrap=1000, ci_level=95):
    # Fit GEV to observed data
    shape, loc, scale = genextreme.fit(data)
    q = genextreme.ppf(1 - 1/return_periods, shape, loc=loc, scale=scale)

    q_boot = np.zeros((n_bootstrap, len(return_periods)))
    n = len(data)

    for i in tqdm(range(n_bootstrap), desc="Parametric Bootstrapping", leave=False):
        # Generate synthetic sample from fitted GEV (parametric bootstrap)
        synthetic_sample = genextreme.rvs(shape, loc=loc, scale=scale, size=n)
        try:
            b_shape, b_loc, b_scale = genextreme.fit(synthetic_sample)
            q_boot[i, :] = genextreme.ppf(1 - 1/return_periods, b_shape, loc=b_loc, scale=b_scale)
        except:
            q_boot[i, :] = np.nan

    # Basic Bootstrap CI: reflect around the original estimate
    alpha = (100 - ci_level) / 100
    lower_percentile = np.nanpercentile(q_boot, 100 * (1 - alpha/2), axis=0)
    upper_percentile = np.nanpercentile(q_boot, 100 * (alpha/2), axis=0)
    lower = 2 * q - lower_percentile
    upper = 2 * q - upper_percentile

    return q, lower, upper

# Read and filter basin list
basin_list = pd.read_csv('data/basins.csv', dtype={'id': str})
# basin_list = basin_list[basin_list['name'].isin(['chepe', 'mardi'])]

# Set up plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 6), sharex=True)
return_periods = np.array([1.1, 2, 5, 10, 25, 50, 75, 100])

# Loop over basins
for ax, (idx, row) in zip(axes.flatten(), basin_list.iterrows()):
    basin_id = row['id']
    basin_name = row['name']

    # Read data
    hist = pd.read_csv(f'output/pp_flow/output_{basin_name}.csv')
    ssp245 = pd.read_csv(f'output/pp_future_flow/output_{basin_name}_ssp245.csv')
    ssp585 = pd.read_csv(f'output/pp_future_flow/output_{basin_name}_ssp585.csv')

    for df in [hist, ssp245, ssp585]:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # Extract annual maxima
    hist_max = get_annual_maxima(hist)
    ssp245_max = get_annual_maxima(ssp245)
    ssp585_max = get_annual_maxima(ssp585)

    # Fit GEV and CI
    q_hist, lower_hist, upper_hist = fit_gev_and_ci(hist_max.values, return_periods)
    q_245, lower_245, upper_245 = fit_gev_and_ci(ssp245_max.values, return_periods)
    q_585, lower_585, upper_585 = fit_gev_and_ci(ssp585_max.values, return_periods)

    # Plot GEV curves and uncertainty bands
    ax.plot(return_periods, q_hist, label='Historical', color=colors['Historical'])
    ax.fill_between(return_periods, lower_hist, upper_hist, color=colors['Historical'], alpha=0.3)

    ax.plot(return_periods, q_245, label='SSP245', color=colors['SSP245'])
    ax.fill_between(return_periods, lower_245, upper_245, color=colors['SSP245'], alpha=0.3)

    ax.plot(return_periods, q_585, label='SSP585', color=colors['SSP585'])
    ax.fill_between(return_periods, lower_585, upper_585, color=colors['SSP585'], alpha=0.3)

    # Plot empirical return levels
    rp_hist, emp_hist = empirical_return_levels(hist_max.values)
    ax.scatter(rp_hist, emp_hist, color=colors['Historical'], marker='o', label='Hist (empirical)', zorder=5, s=12)

    rp_245, emp_245 = empirical_return_levels(ssp245_max.values)
    ax.scatter(rp_245, emp_245, color=colors['SSP245'], marker='s', label='SSP245 (empirical)', zorder=5, s=12)

    rp_585, emp_585 = empirical_return_levels(ssp585_max.values)
    ax.scatter(rp_585, emp_585, color=colors['SSP585'], marker='^', label='SSP585 (empirical)', zorder=5, s=12)

    # Formatting
    ax.set_xscale('log')
    ax.set_title(f'{basin_name.capitalize()} (ID: {basin_id})')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Final plot settings
# x-axis labels and y-axis labels
axes[0, 0].set_xlabel('Return Period (years)')
axes[0, 0].set_ylabel('Streamflow (mm/day)')
axes[0, 1].set_xlabel('Return Period (years)')
axes[0, 1].set_ylabel('Streamflow (mm/day)')
axes[1, 0].set_xlabel('Return Period (years)')
axes[1, 0].set_ylabel('Streamflow (mm/day)')
axes[1, 1].set_xlabel('Return Period (years)')
axes[1, 1].set_ylabel('Streamflow (mm/day)')

plt.tight_layout()
plt.savefig('figures/gev_return_curves_all_basins.jpg', dpi=300)
plt.savefig('figures/inkscape/gev_return_curves_all_basins.svg')
# plt.close()
