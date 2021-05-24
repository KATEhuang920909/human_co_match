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

# model = lightgbm.LGBMClassifier(num_leaves=64,
#                            max_depth=10,
#                            learning_rate=0.1,
#                            n_estimators=1000000,
#                            subsample=0.8,
#                            feature_fraction=0.8,
#                            reg_alpha=0.5,
#                            reg_lambda=0.5,
#                            random_state=2048,
#                            metric='auc')
#
#
#
# lgb_model = model.fit(X_train,
#                       Y_train,
#                       eval_names=['valid'],
#                       eval_set=[(X_val, Y_val)],
#                       verbose=100,
#                       eval_metric='auc',
#                       early_stopping_rounds=100)
#
# pred_val = lgb_model.predict(X_val)
# df_oof = df_train.iloc[val_idx][['RECRUIT_ID', 'PERSON_ID', ycol]].copy()
# df_oof['pred'] = pred_val[:, 1]
# oof.append(df_oof)
#
# pred_test = lgb_model.predict_proba(df_test[feature_names])
# prediction['pred'] += pred_test[:, 1] / kfold.n_splits
#
# df_importance = pd.DataFrame({
#     'column': feature_names,
#     'importance': model.feature_importances_,
# })
# df_importance_list.append(df_importance)
#
