# -*- coding: utf-8 -*-
"""
 Time : 2021/5/20 1:25
 Author : huangkai
 File : tf_idf.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
# representation方式：
"""
tf-idf

"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pickle
from data_preprocess import text_clean


class TF_IDF():
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=100)
        self.dic_path = r"/saved_model/_dic.dic"
        self.tfidf_model_path = r"/saved_model/tfidf.model"
        self.tf_idf_transformer = TfidfTransformer()

    def get_dic(self, word_list):
        dic = self.vectorizer.fit_transform(word_list)
        with open(self.dic_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        return dic

    def get_tf_idf_model(self, word_list):
        tfidf = self.tf_idf_transformer.fit_transform(self.get_dic(word_list))
        with open(self.tfidf_model_path, 'wb') as f:
            pickle.dump(self.tf_idf_transformer, f)

    def encode(self, text):
        text = [text_clean(txt) for txt in text]
        tf_idf_embedding = self.tf_idf_transformer.transform(self.vectorizer.transform(text))
        # print('tf_idf_embedding0>>>>>', tf_idf_embedding)
        tf_idf_embedding = tf_idf_embedding.toarray().sum(axis=0)
        # print ('>>>>', tf_idf_embedding[np.newaxis, :])
        return tf_idf_embedding[np.newaxis, :].astype(float)


if __name__ == '__main__':
    x_train = ['TF-IDF 主要 思想 是', '算法 一个 重要 特点 可以 脱离 语料库 背景',
               '如果 一个 网页 被 很多 其他 网页 链接 说明 网页 重要']
    x_test = ['原始 文本 进行 标记']
    tf_idf = TF_IDF()
    tf_idf.get_tf_idf_model(x_train)
    result = tf_idf.encode(x_test)
    print(result)
