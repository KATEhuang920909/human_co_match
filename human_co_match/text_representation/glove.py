# -*- coding: utf-8 -*-
"""
 Time : 2021/5/21 2:26
 Author : huangkai
 File : glove.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
"""
glove

"""
# 直接加载现成的模型
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input_file = 'F:\\dl-data\\vector\\glove.840B.300d.txt'
word2vec_output_file = 'F:\\dl-data\\vector\\glove.840B.300d.word2vec.txt'


def glove2v2v(glove_path, g2v_path):
    (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
    print(count, '\n', dimensions)


def load_glove(path):
    # 加载模型
    glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
    # 如果希望直接获取某个单词的向量表示，直接以下标方式访问即可
    return glove_model


if __name__ == '__main__':
    pass
    # cat_vec = glove_model['cat']
    # print(cat_vec)
    # # 获得单词frog的最相似向量的词汇
    # print(glove_model.most_similar('frog'))
