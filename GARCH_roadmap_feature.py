# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 11:32:29 2021

@author: Ian.Chen

Using GARCH model to generate predictive values with two dependent variables: roadmap_invoice_qty & rolling_q_invoice_qty
"""

#%% import stats tools
import os
import pandas as pd
import numpy as np
import sys
from arch import arch_model
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt


#get dt data
dt = pd.read_csv('C:/Users/Ian.Chen/Desktop/DT FCST/GARCH_VOLATILITY_FCST/MONTHLY_DT_ROADMAP_FOB.csv')
dt = dt.iloc[:,1:]
dt['time_period'] = pd.DatetimeIndex(dt['time_period'])
# dt = dt.set_index(pd.DatetimeIndex(df['Datetime']))

#acquire complete time period index
orig_dt = dt.copy()

# orig_dt.index = orig_dt['time_period']
orig_first_tp = orig_dt['time_period'].min()
orig_last_tp = orig_dt['time_period'].max()
orig_new_tp = pd.date_range(start=orig_first_tp, end=orig_last_tp, freq='MS')
# orig_new_tp = orig_new_tp.strftime(date_format = "%Y-%m-%d")
orig_new_tp = pd.DataFrame(orig_new_tp, columns=['time_period'])
orig_new_tp = orig_new_tp.set_index('time_period')
final_ttl_df = orig_new_tp.copy()

#Seperate data by Region, roadmap
# dt_roadmap_emea_invoice = dt[dt['region']=='EMEA'].pivot_table(index="time_period", columns="roadmap_group", values="roadmap_invoice_qty")
# dt_roadmap_aap_invoice = dt[dt['region']=='AAP'].pivot(index="time_period", columns="roadmap_group", values="roadmap_invoice_qty")
# dt_roadmap_pa_invoice = dt[dt['region']=='PA'].pivot(index="time_period", columns="roadmap_group", values="roadmap_invoice_qty")
# dt_roadmap_gc_invoice = dt[dt['region']=='GC'].pivot(index="time_period", columns="roadmap_group", values="roadmap_invoice_qty")
# dt_roadmap_china_invoice = dt[dt['region']=='CHINA'].pivot(index="time_period", columns="roadmap_group", values="roadmap_invoice_qty")
# dt_roadmap_ai_invoice = dt[dt['region']=='AI'].pivot(index="time_period", columns="roadmap_group", values="roadmap_invoice_qty")
# dt_segment = dt.pivot_table(index="time_period", columns="platform", values="invoice_qty", aggfunc='first') #used to force agg when appear dup time index data

#Seperate data by Region, roadmap
dt_roadmap_invoice = dt[dt['region']=='CHINA'].pivot_table(index="time_period", columns="roadmap_group", values="roadmap_invoice_qty")
#generate Invoice QTY pct change for roadmaps as dependent df
region = dt_roadmap_invoice.copy()
df = pd.DataFrame()

for i in region.columns:
    empty_df = pd.DataFrame()
    func = []
    region[f'invoice_pct_change_{i}'] = 100*region[i].pct_change()#limit = 1)
    func.append(region[[f'{i}',f'invoice_pct_change_{i}']])
    empty_df = empty_df.append(func)
    
    #process time_period nan gap problem
    test = empty_df.copy()
    test.dropna(inplace = True)
    
    #fill incomplete time period index after dropna
    #create a columns with time period index for further access
    test['time_period'] = test.index
    try:
        first_tp = test['time_period'].min()
        last_tp = test['time_period'].max()
        new_tp = pd.date_range(start=first_tp, end=last_tp, freq='MS')
        # new_tp = new_tp.strftime(date_format = "%Y-%m-%d")
        new_tp = pd.DataFrame(new_tp, columns=['time_period'])
        new_tp = new_tp.set_index('time_period')
        
        #use complete time period index as merging standard and fillna
        complete_tp = new_tp.merge(test, how='left', left_index=True, right_index=True)
        complete_tp = complete_tp.iloc[:,0:2]
        complete_tp.fillna(0, inplace=True)
        
        #replace inf with max + 25%_box plot
        m = complete_tp.loc[complete_tp[f'invoice_pct_change_{i}'] != np.inf, f'invoice_pct_change_{i}'].max() #get max by exclude inf first
        q100, q75, q50, q25 = np.percentile(complete_tp.loc[complete_tp[f'invoice_pct_change_{i}'] != np.inf, f'invoice_pct_change_{i}'], [100, 75, 50 ,25])
        complete_tp[f'invoice_pct_change_{i}'].replace(np.inf,m + q50,inplace=True)
        
        #rolling means
        rolling_windows = complete_tp[f'invoice_pct_change_{i}'].rolling(len(complete_tp.index), min_periods=1)
        complete_tp[f'invoice_pct_change_roll_mean_{i}'] = rolling_windows.mean()
        #create column realised volatility
        complete_tp[f'realised_volatility_{i}'] = complete_tp[f'invoice_pct_change_{i}'] - complete_tp[f'invoice_pct_change_roll_mean_{i}']
        complete_tp[f'realised_volatility_{i}'] = complete_tp[f'realised_volatility_{i}'].abs()
        complete_tp[f'ngtv_realised_volatility_{i}'] = complete_tp[f'realised_volatility_{i}'].abs() * -1
        
        #Merge processed columns back to complete time period index df
        single_roadmap = complete_tp.copy()
        complete_tp = pd.DataFrame()
        
#use complete single roadmap data to run GARCH MODEL(p,q); 
#"p" = the number of autoregressive lags and "q" =  the number of ARCH terms.
#run garch(1,1) #specifying the model and estimating parameters.    
        #using Rolling Window Forecasting
        #Rolling window forecasts use a fixed sample length and then produce one-step from the final observation. 
        #These can be implemented using first_obs and last_obs
        index = single_roadmap.iloc[:,1].index
        start_loc = 0
        end_loc = np.where(index >= f"{single_roadmap.index[2]}")[0].min() #training dataset order
        last = np.where(index >= f"{single_roadmap.index[-1]}")[0].min() # iterate by row for every time stamp fcst
        am = arch_model(single_roadmap.iloc[:,1], vol="Garch", p=1, o=0, q=1, dist="Normal")
        forecasts = {}
        for i in range(last-2): #we extract 3 due to we must skip first 3 time stamp value for min limitation of training dataset
            sys.stdout.write(".")
            sys.stdout.flush()
            res = am.fit(first_obs=0, last_obs=i + end_loc, disp="off")
            temp = res.forecast(horizon=12, reindex=False).variance
            fcast = temp.iloc[2]
            forecasts[fcast.name] = fcast
            fcst_result = pd.DataFrame(forecasts).T
            #pd.set_option('display.float_format', lambda x: '%0.4f' % x)
        # print()
        # print(pd.DataFrame(forecasts).T)
      
        #merge actual with fcst
        sin_rm_fcst = single_roadmap.merge(fcst_result, how='left', left_index=True, right_index=True)
        #change format for shane model
        sin_rm_fcst['roadmap_group'] = sin_rm_fcst.columns[0]
        #change col order
        cols = sin_rm_fcst.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        sin_rm_fcst = sin_rm_fcst[cols]
        sin_rm_fcst.rename(columns={f"{sin_rm_fcst.columns[1]}": "roadmap_invoice_qty", 
                                    f"{sin_rm_fcst.columns[2]}": "roadmap_invoice_qty_pct_change",
                                    f"{sin_rm_fcst.columns[3]}": "roadmap_invoice_qty_pct_change_roll_mean",
                                    f"{sin_rm_fcst.columns[4]}": "roadmap_invoice_qty_pct_change_realised_volatility",
                                    f"{sin_rm_fcst.columns[5]}": "roadmap_invoice_qty_pct_change_ngtv_realised_volatility"
                                    }, inplace = True)

        #Append road_map fcst
        df = df.append(sin_rm_fcst)
        
        # #Merge as final output #No need merge as final now
        # final_df = orig_new_tp.merge(single_roadmap, how='left', left_index=True, right_index=True)
        # final_ttl_df = final_ttl_df.merge(final_df, how='left', left_index=True, right_index=True)
    except:
        pass

df['region'] = 'CHINA'
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

df.to_csv('GARCH_variance(realised)_fcst_china.csv')

#%% Append five region to all
import pandas as pd

new_emea = pd.read_csv('GARCH_variance(realised)_fcst_emea.csv')
new_pa = pd.read_csv('GARCH_variance(realised)_fcst_pa.csv')
new_aap = pd.read_csv('GARCH_variance(realised)_fcst_aap.csv')
new_gc = pd.read_csv('GARCH_variance(realised)_fcst_gc.csv')
new_china = pd.read_csv('GARCH_variance(realised)_fcst_china.csv')

#test unique
new_china.region.nunique()
new_china.region

#start append
test = test.append(new_china, ignore_index=True)
test.to_csv('GARCH_variance(realised)_fcst_all.csv', index = False)

#add ngtv_horizon, sqrt_horizon & sqrt_ngtv_horizon
garch_dt = pd.read_csv('GARCH_variance(realised)_fcst_all.csv') 
#h.1 ~ h.12 locates in columns 8 to 19
garch_dt.columns[8]
garch_dt.columns[19]
#forecast origin final volatility interval with positive and negative
for i in range(12:):
    print(i)
    horizon = 1+i
    columns = 8+i
    garch_dt['ngtv_h.{str(horizon)}'] = garch_dt.apply(lambda row: row[columns] * -1, axis=1)
    
#forecast square root final volatility interval with positive and negative
import math
for i in range(12):
    print(i)
    horizon = 1+i
    columns = 8+i
    garch_dt[f'sqrt_h.{str(horizon)}'] = garch_dt.apply(lambda row: math.sqrt(row[columns]), axis=1)
    
for i in range(12):
    print(i)
    horizon = 1+i
    columns = 8+i
    garch_dt[f'sqrt_ngtv_h.{str(horizon)}'] = garch_dt.apply(lambda row: math.sqrt(row[columns]) * -1, axis=1)
    
garch_dt.to_csv('GARCH_variance(realised)_fcst_final_sqrt.csv')
#%% Implement realised volatility with pct_change = rolling mean with absolute values
#rolling means
rolling_windows = returns.rolling(5030, min_periods=1)
rolling_mean = rolling_windows.mean()

#Merge returns with rolling_mean
returns = returns.to_frame(name='returns')
return_rollmean = returns.merge(rolling_mean, how='left', left_index=True, right_index=True)
return_rollmean = return_rollmean.rename({'Adj Close': 'rolling_mean'}, axis = 1)

#create column realised volatility
return_rollmean['realised_volatility'] = return_rollmean.apply((lambda row: row.returns - row.rolling_mean), axis=1)
return_rollmean['realised_volatility'] = return_rollmean['realised_volatility'].abs()
return_rollmean['ngtv_realised_volatility'] = return_rollmean['realised_volatility'].abs() * -1
return_rollmean.to_csv('S&P500_realised_volatility.csv')
#%%
am = arch_model(single_roadmap.iloc[:,1], vol="Garch", p=1, o=0, q=1, dist="Normal")

index = single_roadmap.iloc[:,1].index
start_loc = 0
end_loc = np.where(index >= "2019-1-1")[0].min()
forecasts = {}
for i in range(20):
    sys.stdout.write(".")
    sys.stdout.flush()
    res = am.fit(first_obs=0, last_obs=i + end_loc, disp="off")
    temp = res.forecast(horizon=3, reindex=False).variance
    fcast = temp.iloc[0]
    forecasts[fcast.name] = fcast
    forecasts_final = pd.DataFrame(forecasts).T
print()
print(pd.DataFrame(forecasts).T)

#%%test for S&P 500 data
from arch import arch_model
import arch.data.sp500
import os
import pandas as pd
import numpy as np
import sys

data = arch.data.sp500.load()
market = data["Adj Close"]
returns = 100 * market.pct_change().dropna()

am = arch_model(returns, vol="Garch", p=1, o=0, q=1, dist="Normal")

#default FCST onlt one step ahead
res = am.fit(update_freq=5)
forecasts = res.forecast(reindex=False)

#print result
print(forecasts.mean.iloc[-3:])
print(forecasts.residual_variance.iloc[-3:])
print(forecasts.variance.iloc[-3:])

#Fixed Window FCST
res = am.fit(last_obs="2000-10-10", update_freq=5)
forecasts = res.forecast(horizon=5, reindex=False)
print(forecasts.variance.dropna().head())
fixed_win_fcst = forecasts.variance.dropna()

#Rolling Window FCST
index = returns.index
start_loc = 0
end_loc = np.where(index >= "1999-1-6")[0].min()
forecasts = {}
for i in range(20):
    sys.stdout.write(".")
    sys.stdout.flush()
    res = am.fit(first_obs=0, last_obs=i + end_loc, disp="off")
    temp = res.forecast(horizon=3, reindex=False).variance
    fcast = temp.iloc[2]
    forecasts[fcast.name] = fcast
    forecasts_final = pd.DataFrame(forecasts).T
print()
print(pd.DataFrame(forecasts).T)

#%%The Simple Garch model
dt_roadmap_emea_invoice.columns
test_df = dt_roadmap_emea_invoice.loc[:,['C Series-19.5','invoice_pct_change_C Series-19.5']]

#Clear nan, (inf, -inf) with any in df
# remove_nan_test_df = test_df[~test_df.isin([np.nan, np.inf, -np.inf]).any(1)] #only remove 

#dropna in column(whether all or thrshold)
remove_nan_test_df_all = test_df.dropna(subset=['C Series-19.5', 'invoice_pct_change_C Series-19.5'], how='all')
remove_nan_test_df_all = remove_nan_test_df_all[~remove_nan_test_df_all.isin([np.nan]).any(1)]
remove_nan_test_df_all.index[1] - remove_nan_test_df_all.index[0]


#filter year
yy = [2018, 2019, 2020]
dt = dt[((dt.year == 2021) & (dt.month < 11)) | ((dt.year == 2017) & (dt.month == 12)) | dt.year.isin(yy)]
dt.reset_index(drop=True, inplace=True)


#run garch(1,1)
model_garch_1_1_test = arch_model(remove_nan_test_df.iloc[:,1], mean = "Constant", vol = "GARCH", p = 1, q = 1)
results_garch_1_1_test = model_garch_1_1_test.fit(update_freq = 5)
results_garch_1_1_test.summary()

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
#%% 移除所有nan後，時間區間內時間值填補，nan再次補0，計算新pct_change()，並以max + 25%_box plot處理inf
test = empty_df.copy()
test.dropna(inplace = True)
#轉row為column
test['time_period'] = test.index
first_tp = test['time_period'].iloc[0]
last_tp = test['time_period'].iloc[-1]
new_tp = pd.date_range(start=first_tp, end=last_tp, freq='MS')
new_tp = new_tp.strftime(date_format = "%Y-%m-%d")
new_tp = pd.DataFrame(new_tp, columns=['time_period'])
new_tp = new_tp.set_index('time_period')

#use complete time period index as merging standard and fillna
complete_tp = new_tp.merge(test, how='left', left_index=True, right_index=True)
complete_tp = complete_tp.iloc[:,0:2]
complete_tp.fillna(0, inplace=True)

#replace inf with max + 25%_box plot
m = complete_tp.loc[complete_tp['invoice_pct_change_Z6'] != np.inf, 'invoice_pct_change_Z6'].max() #get max by exclude inf first
q100, q75, q50, q25 = np.percentile(complete_tp.loc[complete_tp['invoice_pct_change_Z6'] != np.inf, 'invoice_pct_change_Z6'], [100, 75, 50 ,25])
complete_tp['invoice_pct_change_Z6'].replace(np.inf,m + q50,inplace=True)

complete_tp.columns

