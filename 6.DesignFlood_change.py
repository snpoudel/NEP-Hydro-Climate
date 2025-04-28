#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genextreme
import warnings
warnings.filterwarnings('ignore')

basin_list = pd.read_csv('data/basins.csv', dtype={'id': str})
basin_list = basin_list[basin_list['name'].isin(['chepe', 'mardi'])]

basin = 'chepe'
id = 'chepe'
#--historical data--
hist = pd.read_csv(f'output/flow/output_{id}.csv')
hist['date'] = pd.to_datetime(hist['date'], format='%Y-%m-%d')
hist_annual_max = hist.groupby(hist['date'].dt.year)['qsim'].max().reset_index()
#fit gev distribution to annual max series
hist_params = genextreme.fit(hist_annual_max['qsim'])
#get 25, 50, 100 year return period values
hist25, hist50, hist100 = genextreme.ppf([0.96, 0.98, 0.99], *hist_params)


#--future SSP245
ssp245 = pd.read_csv(f'output/future_flow/output_{id}_ssp245.csv')
ssp245['date'] = pd.to_datetime(ssp245['date'], format='%Y-%m-%d')
ssp245_annual_max = ssp245.groupby(ssp245['date'].dt.year)['qsim'].max().reset_index()
#fit gev distribution to annual max series
ssp245_params = genextreme.fit(ssp245_annual_max['qsim'])
#get 25, 50, 100 year return period values
ssp24525, ssp24550, ssp245100 = genextreme.ppf([0.96, 0.98, 0.99], *ssp245_params)

#--future SSP585
ssp585 = pd.read_csv(f'output/future_flow/output_{id}_ssp585.csv')
ssp585['date'] = pd.to_datetime(ssp585['date'], format='%Y-%m-%d')
ssp585_annual_max = ssp585.groupby(ssp585['date'].dt.year)['qsim'].max().reset_index()
#fit gev distribution to annual max series
ssp585_params = genextreme.fit(ssp585_annual_max['qsim'])
#get 25, 50, 100 year return period values
ssp58525, ssp58550, ssp585100 = genextreme.ppf([0.96, 0.98, 0.99], *ssp585_params)

#plot return period values
plt.figure(figsize=(6, 4))
plt.plot([25, 50, 100], [hist25, hist50, hist100], marker='o', label='Historical', color='blue')
plt.plot([25, 50, 100], [ssp24525, ssp24550, ssp245100], marker='o', label='SSP245', color='orange')
plt.plot([25, 50, 100], [ssp58525, ssp58550, ssp585100], marker='o', label='SSP585', color='red')
plt.title(f'{basin} - Return Period Flow Values')
plt.xlabel('Return Period (years)')
plt.ylabel('Flow (mm/s)')
plt.xticks([25, 50, 100])
plt.ylim(0, None)
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig(f'output/return_period/{basin}_return_period.png', dpi=300)
plt.show()

# plt.plot(hist['date'], hist['precip'], label='Historical', color='blue')
# plt.plot(ssp245['date'], ssp245['precip'], label='SSP245', color='orange')
# plt.plot(ssp585['date'], ssp585['precip'], label='SSP585', color='red')