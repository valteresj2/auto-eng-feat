# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 08:00:37 2021

@author: valteresj
"""

import pandas as pd
import numpy as np
from auto_eng_feat import AEF
from random import sample
import lightgbm as lgb
from sklearn import metrics
from scipy import stats

path='C:/Users/valteresj/Documents/Projetos/automl'

dt=pd.read_csv(path+'/bank-full.csv',sep=';')

ind_trn=sample(list(range(dt.shape[0])),int(dt.shape[0]*0.8))
dt_trn=dt.loc[ind_trn,:].copy()
dt_tst=dt.drop(index=ind_trn).copy()

dt_tst=dt_tst.reset_index(drop=True)
dt_trn=dt_trn.reset_index(drop=True)



auto_feat=AEF(dt=dt_trn,n_pol=2,target='y')

get_result_trn=auto_feat.auto_eng_features()

get_result_tst=auto_feat.pred(dt_tst=dt_tst,params=get_result_trn)



######## measure 

param = {'num_leaves': 31,
         'min_data_in_leaf': 32, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.001,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "nthread": 4,
         "verbosity": -1}


dt_train=dt_trn.copy()
dt_test=dt_tst.copy()
target='y'

newdata=get_result_trn['table_final']
dt_test_mod=get_result_tst

trn_idx=sample(list(range(len(dt_train))),int(0.8*len(dt_train)))
val_idx=list(np.where(np.isin(np.array(list(range(len(dt_train)))),np.array(trn_idx))==False)[0])
 
alvo=np.where(dt_train[target]==np.unique(dt_train[target])[0],1,0)

dt_train['pdays']=dt_train['pdays'].fillna(-1)
dt_test['pdays']=dt_test['pdays'].fillna(-1)

trn_data = lgb.Dataset(pd.get_dummies(dt_train.drop(labels=target,axis=1).iloc[trn_idx,:]), label=alvo[trn_idx])
val_data = lgb.Dataset(pd.get_dummies(dt_train.drop(labels=target,axis=1).iloc[val_idx,:]), label=alvo[val_idx])

trn_data1 = lgb.Dataset(pd.get_dummies(newdata.drop(labels=target,axis=1).iloc[trn_idx,:]), label=alvo[trn_idx])
val_data1 = lgb.Dataset(pd.get_dummies(newdata.drop(labels=target,axis=1).iloc[val_idx,:]), label=alvo[val_idx])


num_round = 100000
clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
clf1 = lgb.train(param, trn_data1, num_round, valid_sets = [trn_data1, val_data1], verbose_eval=100, early_stopping_rounds = 200)


est=clf.predict(pd.get_dummies(dt_test.drop(labels=target,axis=1)), num_iteration=clf.best_iteration)  # data origin

est1=clf1.predict(pd.get_dummies(dt_test_mod.drop(labels=target,axis=1)), num_iteration=clf1.best_iteration)  # data eng feat



fpr, tpr, thresholds = metrics.roc_curve(dt_test_mod[target].astype('category').cat.codes, est, pos_label=0)
fpr1, tpr1, thresholds1 = metrics.roc_curve(dt_test_mod[target].astype('category').cat.codes, est1, pos_label=0)


ks_tech=[]
ind_classb=np.where(dt_test_mod[target]==np.unique(dt_test_mod[target])[0])[0]
ind_classm=np.where(dt_test_mod[target]==np.unique(dt_test_mod[target])[1])[0]
ks_orign=stats.ks_2samp(est[ind_classb],est[ind_classm]).statistic
ks_tech.append(ks_orign)

ks_orign=stats.ks_2samp(est1[ind_classb],est1[ind_classm]).statistic
ks_tech.append(ks_orign)
print(ks_tech)
print(metrics.auc(fpr, tpr),metrics.auc(fpr1, tpr1))



