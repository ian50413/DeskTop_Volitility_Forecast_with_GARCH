# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:56:55 2022

@author: Ian.Chen
"""
import pandas as pd


garch_dt = pd.read_csv('GARCH_variance_fcst_all.csv')


#h.1 ~ h.2 locates in columns 6 to 17

#forecast origin final volatility interval with positive and negative
forecasts_final['ngtv_h.3'] = forecasts_final.apply(lambda row: row[2] * -1, axis=1)
#forecast square root final volatility interval with positive and negative
forecasts_final['sqrt_ngtv_h.3'] = forecasts_final.apply(lambda row: math.sqrt(row[2]) * -1, axis=1)

forecasts_final.to_csv('final_complete_sqrt.csv')

garch_dt.columns[17]

import pandas as pd
import math

garch_dt = pd.read_csv('GARCH_variance_fcst_all.csv')


#h.1 ~ h.2 locates in columns 6 to 17

#forecast origin final volatility interval with positive and negative
garch_dt['ngtv_h.12'] = garch_dt.apply(lambda row: row[17] * -1, axis=1)
#forecast square root final volatility interval with positive and negative
garch_dt['sqrt_h.12'] = garch_dt.apply(lambda row: math.sqrt(row[17]), axis=1)
garch_dt['sqrt_ngtv_h.12'] = garch_dt.apply(lambda row: math.sqrt(row[17]) * -1, axis=1)

garch_dt.to_csv('GARCH_variance_fcst_final_sqrt.csv')

garch_dt.columns[17]