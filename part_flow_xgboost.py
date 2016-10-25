# -*- coding:utf-8 -*-

import gc
import numpy as np
import pandas as pd
from pandas import DataFrame

import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, roc_curve
from sklearn.model_selection import KFold, cross_val_score


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

n_train = 1183747
train_part_flow = pd.read_csv('train_production_line_layout.csv', index_col=0)
train_part_flow.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
test_part_flow = pd.read_csv('test_production_line_layout.csv')
test_part_flow.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
magic_features = pd.read_csv('magic_features.csv', names=['Id', 'Response', 'StartTime',
                                                          'mg1', 'mg2', 'mg3', 'mg4'], skiprows=[0])

train_part_flow_r1 = pd.read_csv('train_production_line_layout.csv', index_col=0, nrows=10)
train_part_flow_r1.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)

station_ar = train_part_flow_r1.columns.values[1:-1]
station_se = pd.Series(station_ar)

start_station_dict = dict(zip(station_ar, np.zeros(len(station_ar))))

part_flow = train_part_flow.append(test_part_flow)
train_test = part_flow.merge(magic_features, on='Id')
del train_part_flow, test_part_flow, part_flow, magic_features
gc.collect()

# print train_test.shape
# print train_test.columns
# print np.unique(train_test['Response_x'][:n_train] == train_test['Response_y'][:n_train])
train_test.drop('Response_y', axis=1, inplace=True)
train_test.rename(columns={'Response_x': 'Response'}, inplace=True)
train_test.to_csv('part_flow_and_magic_features.csv')
# print type(train_test)
# print train_test.shape
# print train_test.head(1)
features = list(station_ar) + ['mg1', 'mg2', 'mg3', 'mg4']
# print features

# # # train xgboost tree -------------
# xgb_params = {
#     'seed': 0,
#     'colsample_bytree': 0.7,
#     'silent': 1,
#     'subsample': 0.7,
#     'learning_rate': 0.1,
#     'objective': 'binary:logistic',
#     'max_depth': 4,
#     'num_parallel_tree': 1,
#     'min_child_weight': 2,
#     'eval_metric': 'auc',
#     'base_score': 0.005
# }
#
# X_train = train_test[features].iloc[:n_train]
# y_train = train_test['Response'].iloc[:n_train]
# X_test = train_test[features].iloc[n_train:]
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test)

# res = xgb.cv(xgb_params, dtrain, num_boost_round=10, nfold=4, seed=0, stratified=True,
#              early_stopping_rounds=1, verbose_eval=1, show_stdv=True)

# bst = xgb.train(xgb_params, dtrain, 10)
# preds = bst.predict(dtest)

# # find the best threshold -------------------
# fpr, tpr, thresholds = roc_curve(y_train, preds, pos_label=1)
# threshold_var = []
# for num in np.arange(0, max(preds), max(preds)/100.):
#     threshold_var.append([num, matthews_corrcoef(y_train, preds > num)])
# threshold_var = pd.DataFrame(threshold_var, columns=['thresholds', 'mcc'])
# i_max_mcc = threshold_var['mcc'].idxmax()
# best_threshold = threshold_var['thresholds'][i_max_mcc]
# print best_threshold, threshold_var['mcc'].max()

# # predict ---------------------
# print len(preds)
# print len(X_test)
# result = pd.DataFrame((preds > 0.2767).astype(int), index=train_test['Id'][n_train:])
# result.to_csv('submission_1025.csv', header=['Response'])

# bst = xgb.train(xgb_params, dtrain, 10)
# test_part_flow = pd.read_csv('test_production_line_layout.csv')
# test_part_flow.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
# X_test = test_part_flow.iloc[:, 1:]
# y_test = bst.predict(X_test)


# prior = np.sum(Y) / (1.*len(Y))
# clf = xgb.XGBClassifier(seed=0, silent=1, learning_rate=0.1, objective='binary:logistic',
# max_depth=4, min_child_weight=2, base_score=prior)
# clf.fit(X, Y)
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test)
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
