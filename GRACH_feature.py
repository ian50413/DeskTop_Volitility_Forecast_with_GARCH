# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:41:40 2021

@author: Ian.Chen

This file aims to create feature by using GARCH Stats time seires predictive model
"""

#%% import stats tools
import os
import pandas as pd
import numpy as np
import random
import itertools
from arch import arch_model
from scipy.stats import shapiro
from scipy.stats import probplot
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight') 
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
%matplotlib inline

dt = pd.read_csv('C:/Users/Ian.Chen/Desktop/DT FCST/1224_Invoice_qty/MONTHLY_DT_FOB_WITH_ARIMA_ROADMAP_LAG.csv')
dt = dt.iloc[:,1:]

list(dt['region'].unique())

#Seperate Regions to exclude duplicate time_period when setting index 
dt_segment_emea = dt[dt['region']=='EMEA'].pivot(index="time_period", columns="platform", values="invoice_qty")
dt_segment_aap = dt[dt['region']=='AAP'].pivot(index="time_period", columns="platform", values="invoice_qty")
dt_segment_pa = dt[dt['region']=='PA'].pivot(index="time_period", columns="platform", values="invoice_qty")
dt_segment_gc = dt[dt['region']=='GC'].pivot(index="time_period", columns="platform", values="invoice_qty")
dt_segment_china = dt[dt['region']=='CHINA'].pivot(index="time_period", columns="platform", values="invoice_qty")
dt_segment_ai = dt[dt['region']=='AI'].pivot(index="time_period", columns="platform", values="invoice_qty")
# dt_segment = dt.pivot_table(index="time_period", columns="platform", values="invoice_qty", aggfunc='first') #used to force agg when appear dup time index data

#generate Invoice QTY pct change for every platform
for i in dt_segment_emea.columns:
    dt_segment_emea[f"pct_change_{i}"] = 100*dt_segment_emea[i].pct_change()

#plot Invoice QTY pct change for every platform
for i in dt_segment_emea.columns:
    try:
        pf = dt_segment_emea[[f'{i}', f'pct_change_{i}']]
        pf.dropna(inplace=True)
        fig_ori = pf[f'{i}'].plot(figsize=(10, 5), title=f'{i} Invoice QTY').get_figure()
        fig_ori.savefig('C:/Users/Ian.Chen/Desktop/DT FCST/1224_Invoice_qty/' + f"Invoice_QTY/{i}_Invoice_QTY.png")
        fig_ori.clear()
        fig_pct = pf[f'pct_change_{i}'].plot(figsize=(10, 5), title=f'{i} Percent Change in Invoice QTY').get_figure()
        fig_pct.savefig('C:/Users/Ian.Chen/Desktop/DT FCST/1224_Invoice_qty/' + f"pct_change_fig/{i}_pct_change.png")
        fig_pct.clear()
    except:
        pass

#12/28 Due to data incomplete
#%%
dt_emea_sum = dt[dt['region']=='EMEA'].pivot(index="time_period", columns="platform", values="invoice_qty")

        #plot acf & pacf
        # acf = plot_acf(pf[f'pct_change_{i}'], lags=30)
        # pacf = plot_pacf(pf[f'pct_change_{i}'], lags=30)
        # acf.suptitle(f'{i} Percent Change Autocorrelation', fontsize=20)
        # acf.set_figheight(5)
        # acf.set_figwidth(15)
        # fig_acf = acf.get_figure()
        # fig_acf.savefig('C:/Users/Ian.Chen/Desktop/DT FCST/1224_Invoice_qty/' + f"pct_change_acf/{i}_pct_change_acf.png")
        # pacf.suptitle(f'{i} Percent Change Partial Autocorrelation', fontsize=20)
        # pacf.set_figheight(5)
        # pacf.set_figwidth(15)
        # fig_pacf = pacf.get_figure()
        # fig_pacf.savefig('C:/Users/Ian.Chen/Desktop/DT FCST/1224_Invoice_qty/' + f"pct_change_pacf/{i}_pct_change_pacf.png")
        
dt_AZ20_730 = dt_segment_emea[['AZ20-730', 'pct_change_AZ20-730']]
dt_AZ20_730.dropna(inplace=True)
dt_AZ20_730['AZ20-730'].plot(figsize=(10, 5), title='AZ20-730 Invoice QTY')
plt.show()
fig = dt_AZ20_730['pct_change_AZ20-730'].plot(figsize=(10, 5), title='AZ20-730 Percent Change in Invoice QTY').get_figure()
fig.savefig('pct_change_AZ20-730.png')
fig.clear()
row_num = len(dt_AZ20_730.index)
acf = plot_acf(dt_AZ20_730['pct_change_AZ20-730'], lags= (row_num - 1) )
pacf = plot_pacf(dt_AZ20_730['pct_change_AZ20-730'], lags= (row_num / 2 -1))
acf.suptitle('AZ20-730 Percent Change Autocorrelation and Partial Autocorrelation', fontsize=20)
acf.set_figheight(5)
acf.set_figwidth(15)
fig_acf = acf.get_figure()
fig_acf.savefig('C:/Users/Ian.Chen/Desktop/DT FCST/1224_Invoice_qty/' + f"pct_change_acf/{i}_pct_change_acf.png")

pacf.set_figheight(5)
pacf.set_figwidth(15)
plt.show()

#%%The Simple Garch model

#Clear nan, inf, -inf with any in df
dt_AZ20_730 = dt_AZ20_730[~dt_AZ20_730.isin([np.nan, np.inf, -np.inf]).any(1)]
#run garch(1,1)
model_garch_1_1_AZ20730 = arch_model(dt_AZ20_730.iloc[:,1], mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_garch_1_1_AZ20730 = model_garch_1_1_AZ20730.fit(update_freq = 5)
results_garch_1_1_AZ20730.summary()

#run garch(1,2) Higher-Lag GARCH Models
model_garch_1_2_AZ20730 = arch_model(dt_AZ20_730.iloc[:,1], mean = "Constant", vol = "GARCH", p = 1, q = 2)
results_garch_1_2_AZ20730 = model_garch_1_2_AZ20730.fit(update_freq = 5)
results_garch_1_2_AZ20730.summary()

#run garch(2,1)
model_garch_2_1_AZ20730 = arch_model(dt_AZ20_730.iloc[:,1], mean = "Constant", vol = "GARCH", p = 2, q = 1)
results_garch_2_1_AZ20730 = model_garch_2_1_AZ20730.fit(update_freq = 5)
results_garch_2_1_AZ20730.summary()

#run garch(2,2)
model_garch_2_2_AZ20730 = arch_model(dt_AZ20_730.iloc[:,1], mean = "Constant", vol = "GARCH", p = 2, q = 2)
results_garch_2_2_AZ20730 = model_garch_2_2_AZ20730.fit(update_freq = 5)
results_garch_2_2_AZ20730.summary()

#FORECASTING
#Building Prediction Data
dt_AZ20_730["Predictions"] = results_garch_1_1_AZ20730.forecast().residual_variance.loc[dt_AZ20_730.index]

forecasts = results_garch_1_1_AZ20730.forecast(reindex=False)
print(forecasts.mean.iloc[:])
print(forecasts.residual_variance.iloc[:])
print(forecasts.variance.iloc[:])

forecasts = results_garch_1_1_AZ20730.forecast(horizon=5, reindex=True)
print(forecasts.residual_variance.iloc[:])


dt['roadmap_group'].nunique()
