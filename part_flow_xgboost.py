# -*- coding:utf-8 -*-

import gc
import numpy as np
import pandas as pd
from pandas import DataFrame

import xgboost as xgb
from sklearn.cross_validation import KFold, cross_val_score


# # read production_line_layout part files and append them to a whole file-------------
# train_startion = DataFrame()
# for k in range(12):
#     df = pd.read_csv('test_production_line_layout_{0}.csv'.format(str(k)),
#                      sep=' ', dtype=np.uint16).transpose()
#     train_startion = train_startion.append(df)
#     gc.collect()
#     print(len(df))

# # read train_numeric response and merge it to DataFram train_startion-----------
# Train_Num = pd.read_csv('/media/ubuntu/新加卷2/Kaggle/Featured/Bosch\
# /train_numeric.csv', dtype={'Id':np.float64, 'Response':np.uint8}, usecols=['Id', 'Response'])
# print Train_Num.head()
# train_startion = pd.read_csv('production_line_layout.csv')
# train_startion.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
# print train_startion.head()
# train_startion = train_startion.merge(Train_Num, on='Id')
# print train_startion.head()
# train_startion.to_csv('test_production_line_layout.csv')
#

train_part_flow = pd.read_csv('train_production_line_layout.csv', index_col=0, chunksize=100000)
train_part_flow_r1 = pd.read_csv('train_production_line_layout.csv', index_col=0, nrows=10)
train_part_flow_r1.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)

station_ar = train_part_flow_r1.columns.values[1:-1]
station_se = pd.Series(station_ar)

start_station_dict = dict(zip(station_ar, np.zeros(len(station_ar))))




# train xgboost tree -------------
# X = train_part_flow.iloc[:, 1:-1]
# Y = train_part_flow.iloc[:, -1]
# prior = np.sum(Y) / (1.*len(Y))
# clf = xgb.XGBClassifier(seed=0, silent=1, learning_rate=0.1, objective='binary:logistic',
# max_depth=4, min_child_weight=2, base_score=prior)
# clf.fit(X, Y)

# kf = KFold(n=len(X), n_folds=5, shuffle=False, random_state=False)
# for iTrain, iTest in kf:
#     xTrain, yTrain = X.iloc[iTrain, :], Y.iloc[iTrain]
#     xTest,  yTest  = X.iloc[iTest, :],  Y.iloc[iTest]
#
#     clf.fit(xTrain, yTrain)
#     pred = clf.predict(xTest)
#     print np.sum(pred)

#     print np.sum(pred)
# print cross_val_score(clf, X, Y, cv=5, scoring='roc_auc')
#
# # predict using xgboost tree-----------------
# test_part_flow = pd.read_csv('test_production_line_layout.csv')
# test_part_flow.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
# X_test = test_part_flow.iloc[:, 1:]
# y_test = clf.predict(X_test)
# print np.sum(y_test)
#
# # write predict result to csv, according to sample format-----------------
# with open('submission.csv', 'w') as f:
#     f.write('Id,Response\n')
#     for k in range(0, len(y_test)):
#         f.write(str(test_part_flow['Id'][k]))
#         f.write(',')
#         f.write(str(y_test[k]))
#         f.write('\n')
# f.close()


# train_part_flow = train_part_flow.sort_values(list(station_ar), ascending=False).reset_index()


# def find_start_station(k):
#     one_part_flow = train_part_flow_part[station_ar].iloc[k, :].values.astype(bool)
#     try:
#         start_station = station_se[one_part_flow].iloc[0]
#     except:
#         IndexError
#         return
#     start_station_dict[start_station] += 1

# count = 0
# for train_part_flow_part in train_part_flow:
#     print count
#     for k in range(0, len(train_part_flow_part)):
#         # print k
#         find_start_station(k)
#     count += 1
# pd.Series(start_station_dict).to_csv('train_start_station_count.csv')

# print train_part_flow_r1 == 1
