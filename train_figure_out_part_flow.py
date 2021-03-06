# -*- coding:utf-8 -*-

import re
import collections
import numpy as np
import pandas as pd
from pandas import DataFrame
# import matplotlib.pyplot as plt
# import seaborn as sns


# read train data
def read_data(nlines = 2, partsize = None):
    # read column names
    Num_Cols = pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch\
/train_numeric.csv', nrows=1, dtype=np.float16).columns
    Cate_Cols = pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch\
/train_categorical.csv', nrows=1, dtype=str).columns
    Date_Cols = pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch\
/train_date.csv', nrows=1, dtype=np.float16).columns

    Num_Cols_flo = Num_Cols[1:-1]
    Cate_Cols_flo = Cate_Cols[1:]
    Date_Cols_flo = Date_Cols[1:]

    Num_dtype_map = {'Id': np.float64, 'Response':np.uint8}
    Cate_dtype_map = {'Id': np.float64}
    Date_dtype_map = {'Id': np.float64}

    Num_dtype_map.update({feature: np.float16 for feature in Num_Cols_flo})
    Cate_dtype_map.update({feature: str for feature in Cate_Cols_flo})
    Date_dtype_map.update({feature: np.float16 for feature in Date_Cols_flo})

    Train_Num = pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch\
/train_numeric.csv', nrows=nlines, dtype=Num_dtype_map, chunksize=partsize)
#     Train_Cate_Part = pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch\
# /train_categorical.csv', nrows=nlines, dtype=Cate_dtype_map, chunksize=partsize)
#     Train_Date_Part = pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch\
# /train_date.csv', nrows=nlines, dtype=Date_dtype_map, chunksize=partsize)

    return Train_Num


# line station and feature numerical value
def line_station_num(Num_Cols):
    Num_list = []
    pattern = re.compile('L\d+_S(\d+)_F(\d+)')
    for col in Num_Cols[1: -1]:
        lsf = re.match(pattern, col)
        Num_list.append([int(lsf.group(1)[:]), lsf.group(2)])
    Num_df = DataFrame(Num_list, columns=['Station', 'F'])

    return Num_df


# line station and date
def line_station_cate(Cate_Cols):
    Cate_list = []
    pattern = re.compile('L\d+_S(\d+)_F(\d+)')
    for col in Cate_Cols[1:]:
        lsd = re.match(pattern, col)
        Cate_list.append([int(lsd.group(1)[:]), lsd.group(2)])
    Cate_df = DataFrame(Cate_list, columns=['Station', 'F'])
    # Cate_df.sort_values(by=['Line_Station', 'F'], inplace=True)

    return Cate_df


# line station and feature categorical value
def line_station_date(Date_Cols):
    Date_list = []
    pattern = re.compile('L\d+_S(\d+)_D(\d+)')
    for col in Date_Cols[1:]:
        lsd = re.match(pattern, col)
        Date_list.append([int(lsd.group(1)[:]), lsd.group(2)])
    Date_df = DataFrame(Date_list, columns=['Station', 'D'])

    return Date_df


# Feature measured at which day
def f_measured_in_d(Date_df, Num_df, Num_df_Station):
    MT_list = []
    for Station in list(Num_df_Station):
        lsd = Date_df[Date_df['Station'] == Station].reset_index()
        lsf = Num_df[Num_df['Station'] == Station].reset_index()

        mask = lsf['F'] < lsd['D'][0]
        if mask.sum() > 0:
            F_in_D = lsf['F'][mask].values
            temp = list([Station, lsd['D'][0], F_in_D, len(F_in_D)])
            MT_list.append(temp)
        for k in range(len(lsd) - 1):
            # ix = np.where(np.logical_and(lsf['F'] > lsd['D'][k], lsf['F'] < lsd['D'][k + 1]))
            mask = (lsf['F'] > lsd['D'][k]) & (lsf['F'] < lsd['D'][k + 1])
            if mask.sum() > 0:
                F_in_D = lsf['F'][mask].values
                temp = list([Station, lsd['D'][k + 1], F_in_D, len(F_in_D)])
                MT_list.append(temp)
    MT_df = DataFrame(MT_list, columns=['Station', 'D', 'F', 'F_cnt'])

    return MT_df


def gen_part_flow(Train_Num_Part, part_station_by_col):
    part_flow = []
    for k in range(1, len(part_station_by_col)):
        station = part_station_by_col.index[k]
        sta_ind_l = part_station_by_col.iloc[k-1, 0] + 1
        sta_ind_r = part_station_by_col.iloc[k, 0] + 1

        same_sta = np.float16(Train_Num_Part.iloc[:, sta_ind_l:sta_ind_r].sum(axis=1).fillna(0))
        print Train_Num_Part.iloc[:, sta_ind_l:sta_ind_r]
        print '------'
        same_sta[same_sta != 0] = int(1)
        same_sta = np.uint8(same_sta)
        same_sta = list(same_sta)
        same_sta.insert(0, int(station))
        part_flow.append(same_sta)

    return part_flow


def save_part_flow(parts_list, part_flow, count):
    f = open('production_line_layout_' + str(count) + '.csv', 'w')

    for x in parts_list[:-1]:
        f.write('%d ' % x)
    f.write('%d' % parts_list[-1])
    f.write('\n')

    for x in part_flow:
        for y in x[:-1]:
            f.write('%d ' % y)
        f.write('%d' % x[-1])
        f.write('\n')

    f.close()

Train_Num = read_data(1, None)
part_station_by_col = pd.read_csv('train_part_station_by_col.csv', index_col=0)

count = 0
# for df in Train_Num:
#     parts_list = list(df['Id'])
#     part_flow = gen_part_flow(df, part_station_by_col)
#     save_part_flow(parts_list, part_flow, count)
#
#     count += 1

parts_list = list(Train_Num['Id'])
part_flow = gen_part_flow(Train_Num, part_station_by_col)
# save_part_flow(parts_list, part_flow, count)
print(Train_Num.shape)

count += 1