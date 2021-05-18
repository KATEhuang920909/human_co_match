# -*- coding: utf-8 -*-
"""
 Time : 2021/5/15 8:41
 Author : huangkai
 File : data_preprocess.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
from string import punctuation as eg_punc
from zhon.hanzi import punctuation as ch_punc
# 数据清洗
import re
import jieba

jieba.load_userdict(r"D:\learning\competition\人岗匹配\human_co_match\human_co_match\data\dict.txt")


def text_clean(txt):
    result = re.split(r"[{}]".format(eg_punc + ch_punc), txt)
    return ''.join(result)


def text_tokenizer(txt):
    sentence = text_clean(txt)
    sentence = jieba.lcut(sentence)
    return sentence


if __name__ == '__main__':
    # print(text_tokenizer("/普通话二级甲，普通话二甲，普通话二级乙，普通话二乙，"))
    print(text_clean("【园艺学】"))