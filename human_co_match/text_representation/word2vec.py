# -*- coding: utf-8 -*-
"""
 Time : 2021/5/21 1:50
 Author : huangkai
 File : word2vec.py
 Software: PyCharm
 mail:18707125049@163.com

"""

"""
word2vec mean

"""
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from data_preprocess import text_clean
import jieba


class WORD2VEC():

    def word2vec_train(self, data, pre_train_path=None, need_finetune=False):  # 全量训练/微调
        """
        data形状：[tokensize_sentence1,tokensize_sentence2...]
        1.是否有pre_model
        2.是否需要fine tune
        :param data:已经分好词了
        :return:
        """

        if pre_train_path:
            self.model = Word2Vec.load(pre_train_path)

            if need_finetune:
                self.model.train(data, total_examples=1, epochs=1)
                self.model.save('../saved_model/word2vec.model')

        else:
            self.model = Word2Vec(data, size=256, window=5, min_count=1, workers=4)
            self.model.save('../saved_model/word2vec.model')
            # np.save('../pre_train_model/word2vec.npy',model,allow_pickle=True)
            self.model.wv.save_word2vec_format('../pre_train_model/word2vec.vector', binary=False)

    def encode(self, txt):
        txt_vec = []
        txt = text_clean(txt)
        for word in txt:
            word_vec = self.model[word]
            txt_vec.append(word_vec)
        return txt_vec

    def w2v_mean(self, txt_vec):
        txt_vec = sum(txt_vec) / len(txt_vec)
        return txt_vec


if __name__ == '__main__':
    data_loader = input_helpers.InputHelper()
    data_loader.load_file()
    print(data_loader.x_train[0])
    print(data_loader.x_test[0])
    data_tokenize = data_loader.data_token(data_loader.x_train + data_loader.x_test)
    print(data_tokenize[0])
    word2vec_train(data_tokenize)
