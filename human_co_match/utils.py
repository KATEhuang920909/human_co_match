# -*- coding: utf-8 -*-
"""
 Time : 2021/5/15 8:40
 Author : huangkai
 File : utils.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
import pandas as pd
from data_preprocess import text_clean
from embedding import bert_feature_extract
from sklearn.model_selection import train_test_split


def load_data(path):
    train_data = pd.read_csv(path + "train_data.csv")
    test_data = pd.read_csv(path + "test_data.csv")
    valid_data = pd.read_csv(path + "valid_data.csv")
    return train_data, valid_data, test_data



def submission(path, model, data, name):
    data["预测打分"] = model.predict(data[data.columns[:-1]].values)
    test_data = data.sort_values("预测打分", ascending=False, ignore_index=True)
    test_data["预测"] = 0
    test_data.loc[:int(0.15 * len(test_data)), ["预测"]] = 1

    submit = test_data[["岗位编号", "求职者编号", "预测"]]
    submit.columns = ["RECRUIT_ID", "PERSON_ID", "LABEL"]
    submit.to_csv(path + name + ".csv", index=False)





if __name__ == '__main__':
    txt1='工程管理'
    txt2='工商管理'
    print(Jaccrad(txt1,txt2))