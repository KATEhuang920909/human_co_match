# -*- coding: utf-8 -*-
"""
 Time : 2021/5/15 8:40
 Author : huangkai
 File : lgb_model.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
import lightgbm
from sklearn.metrics import f1_score
import numpy as np

params = {"objective": "binary",
          "learning_rate": 0.05,
          "max_depth": 6,
          "num_leaves": 32,
          "verbose": -1,
          "bagging_fraction": 0.8,
          "feature_fraction": 0.9,
          'subsample': 0.85,
          'bagging_freq': 1,
          'random_state': 2048}


def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat)  # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


def lgb_model_for_upload(data, label):
    model = lightgbm.train(train_set=lightgbm.Dataset(data.values, label.values),
                           num_boost_round=500,
                           params=params)

    return model


def lgb_model_for_offline_test(train, train_label, valid, valid_label):
    train_set = lightgbm.Dataset(train.values, label=train_label)
    val_set = lightgbm.Dataset(valid.values, label=valid_label)

    model = lightgbm.train(train_set=train_set,
                           valid_sets=[val_set],
                           num_boost_round=3000,
                           params=params,
                           feval=lgb_f1_score,
                           early_stopping_rounds=100)
    return model
