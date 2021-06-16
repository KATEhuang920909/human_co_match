# -*- coding: utf-8 -*-
"""
 Time : 2021/6/16 21:28
 Author : huangkai
 File : embedding.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""

import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array

config_path = r'D:\learning\project\chinese_L-12_H-768_A-12\bert_config.json'
checkpoint_path = r'D:\learning\project\chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = r'D:\learning\project\chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

"""
输出：
[[[-0.63251007  0.2030236   0.07936534 ...  0.49122632 -0.20493352
    0.2575253 ]
  [-0.7588351   0.09651865  1.0718756  ... -0.6109694   0.04312154
    0.03881441]
  [ 0.5477043  -0.792117    0.44435206 ...  0.42449304  0.41105673
    0.08222899]
  [-0.2924238   0.6052722   0.49968526 ...  0.8604137  -0.6533166
    0.5369075 ]
  [-0.7473459   0.49431565  0.7185162  ...  0.3848612  -0.74090636
    0.39056838]
  [-0.8741375  -0.21650358  1.338839   ...  0.5816864  -0.4373226
    0.56181806]]]
"""


def saved_model(path):
    config_path = r'D:\learning\project\chinese_L-12_H-768_A-12\bert_config.json'
    checkpoint_path = r'D:\learning\project\chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = r'D:\learning\project\chinese_L-12_H-768_A-12/vocab.txt'
    pass


def bert_feature_extract(txt):
    # 编码测试
    token_ids, segment_ids = tokenizer.encode(txt)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # print()
    return model.predict([token_ids, segment_ids])[0][0]

if __name__ == '__main__':
    print(bert_feature_extract("语言模型"))
