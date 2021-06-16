# -*- coding: utf-8 -*-
"""
 Time : 2021/6/16 21:38
 Author : huangkai
 File : bert_lstm.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
# 自我描述、求职意向、工作经历、证书、项目经历、岗位要求
# 768*5=3840维特征
from utils import load_data, data2vec
import pandas as pd
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
train_data, valid_data, test_data = load_data(
    r"D:\learning\competition\人岗匹配\human_co_match\human_co_match\data\text_data\\")
data = pd.concat([train_data, valid_data, test_data])
print(data.columns)
# txt2vec & padding
person_list = data2vec(data, "求职者文本内容")
intent_list = data2vec(data, "投递意向文本内容")
work_exp_list = data2vec(data, "工作经历文本内容")
cert_list = data2vec(data, "证书文本内容")
project_exp_list = data2vec(data, "项目经历文本内容")
job_desc_list = data2vec(data, "岗位文本内容")
y_list = data["标签"].values

# data split
train_idx, valid_idx = train_data.shape[0], train_data.shape[0] + valid_data.shape[0]
person_train, person_eval, person_test = person_list[:train_idx], \
                                         person_list[train_idx:valid_idx], \
                                         person_list[valid_idx:]
intent_train, intent_eval, intent_test = intent_list[:train_idx], \
                                         intent_list[train_idx:valid_idx], \
                                         intent_list[valid_idx:]
work_exp_train, work_exp_eval, work_exp_test = work_exp_list[:train_idx], \
                                               work_exp_list[train_idx:valid_idx], \
                                               work_exp_list[valid_idx:]
cert_train, cert_eval, cert_test = cert_list[:train_idx], \
                                   cert_list[train_idx:valid_idx], \
                                   cert_list[valid_idx:]
project_exp_train, project_exp_eval, project_exp_test = project_exp_list[:train_idx], \
                                                        project_exp_list[train_idx:valid_idx], \
                                                        project_exp_list[valid_idx:]
job_desc_train, job_desc_eval, job_desc_test = job_desc_list[:train_idx], \
                                               job_desc_list[train_idx:valid_idx], \
                                               job_desc_list[valid_idx:]
y_train, y_eval, y_test = y_list[:train_idx], \
                          y_list[train_idx:valid_idx], \
                          y_list[valid_idx:]



class Metrics(Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return


model = Sequential()
x_train = np.concatenate((person_train, intent_train, work_exp_train, cert_train, project_exp_train, job_desc_train),
                         axis=1)
x_eval = np.concatenate((person_eval, intent_eval, work_exp_eval, cert_eval, project_exp_eval, job_desc_eval),
                        axis=1)
x_test = np.concatenate((person_test, intent_test, work_exp_test, cert_test, project_exp_test, job_desc_test),
                        axis=1)
model.add(LSTM(output_dim=200, activation='tanh', input_shape=(-1, 768 * 5)))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=256,
                    callbacks=[Metrics(valid_data=(x_eval, y_eval))],
                    validation_data=(x_eval, y_eval),
                    verbose=1)

print("Evaluate...")
score = f1_score(x_test, y_test)
print(score)
