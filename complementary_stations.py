# -*- coding:utf-8 -*-

import re
import collections
import numpy as np
import pandas as pd
from pandas import DataFrame


def get_station_col_loc(station, part_station_by_col):
    station = part_station_by_col.index.get_loc(station)
    sta_ind_l = part_station_by_col.iloc[station-1, 0] + 1
    sta_ind_r = part_station_by_col.iloc[station, 0] + 1
    return sta_ind_l, sta_ind_r

def cal_sta_meas_mean(usecolumns):
    return pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch/train_numeric.csv',
                       usecols=usecolumns, dtype=np.float16).mean()

Num_Cols = pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch\
/train_numeric.csv', nrows=1, dtype=np.float16).columns.values

part_station_by_col = pd.read_csv('part_station_by_col.csv', index_col=0)

station = [21, 22, 23]
mean_result = []
for sta in station:
    sta_ind_l, sta_ind_r = get_station_col_loc(str(sta), part_station_by_col)
    usecolumns = Num_Cols[sta_ind_l:sta_ind_r]
    print cal_sta_meas_mean(usecolumns)

