# -*- coding: utf-8 -*-
"""
 Time : 2021/5/21 2:23
 Author : huangkai
 File : doc2vec.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
"""
doc2vec 特征
"""

# python example to train doc2vec model (with or without pre-trained word embeddings)

import gensim.models as gm
import logging
import codecs
import numpy as np
from data_preprocess import text_clean

# doc2vec 参数


# min_count = 1  # 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
#
# negative_size = 5  # 如果>0,则会采用negativesampling，用于设置多少个noise words（一般是5-20）
# train_epoch = 100  # 迭代次数
# dm = 0  # 0 = dbow; 1 = dmpv
# worker_count = 1  # 用于控制训练的并行数
#
# # pretrained word embeddings
# pretrained_emb = "toy_data/pretrained_word_embeddings.txt"  # None if use without pretrained embeddings
#
# # 输入语料库
# train_corpus = "toy_data/wiki_en.txt"
#
# # 模型输出
# save_model_name = 'wiki_en_doc2vec.model'
# saved_path = "toy_data/model/wiki_en_doc2vec.model.bin"

# 获取日志信息
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class DOC2VEC():
    def __init__(self):

        # 超参
        self.start_alpha = 0.01
        self.infer_epoch = 1000
        self.vector_size = 256  # 词向量长度，默认为100
        self.window_size = 15  # 窗口大小，表示当前词与预测词在一个句子中的最大距离是多少
        self.sampling_threshold = 1e-5  # 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)
        self.worker_count = 1  # 用于控制训练的并行数
        self.dm = 0
        self.negative_size = 5
        self.min_count = 1
        self.train_epoch = 100

    def doc2vec_train(self, data, pre_train_path=None, need_finetune=False):  # 全量训练/微调
        """
        data形状：[tokensize_sentence1,tokensize_sentence2...]
        1.是否有pre_model
        2.是否需要fine tune
        :param data:已经分好词了
        :return:
        """
        docs = gm.doc2vec.TaggedLineDocument(data)  # 加载语料
        if pre_train_path:
            self.model = gm.Doc2Vec.load(pre_train_path)

            if need_finetune:
                self.model.train(docs, total_examples=1, epochs=1)
                self.model.save('../saved_model/doc2vec.model')

        else:
            self.model = gm.Doc2Vec(docs, size=self.vector_size, window=self.window_size, min_count=self.min_count,
                                    sample=self.sampling_threshold,
                                    workers=self.worker_count, hs=0, dm=self.dm, negative=self.negative_size, dbow_words=1,
                                    dm_concat=1,
                                    iter=self.train_epoch)
            self.model.save('../saved_model/doc2vec.model')
            # np.save('../pre_train_model/word2vec.npy',model,allow_pickle=True)

    def encode(self, sentence):

        return self.model.infer_vector(sentence, alpha=self.start_alpha, steps=self.infer_epoch)

# 训练 doc2vec 模型

# model = gm.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold,
#                    workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1,
#                    iter=train_epoch)

# # 保存模型
# model.save(saved_path)
#
# model = "toy_data/model/wiki_en_doc2vec.model.bin"
# test_docs = "toy_data/test.txt"  # test.txt为需要向量化的文本
# output_file = "toy_data/test_vector.txt"  # 得到测试文本的每一行的向量表示
#

#
# # 加载模型
# m = gm.Doc2Vec.load(model)
# test_docs = [x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines()]
#
# # infer test vectors
# output = open(output_file, "w")
# for d in test_docs:
#     output.write(" ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n")
# output.flush()
# output.close()
# # print(len(test_docs)) #测试文本的行数
#
# print(m.most_similar("party", topn=5))  # 找到与party单词最相近的前5个
#
# # 保存为numpy形式
# test_vector = np.loadtxt('toy_data/test_vector.txt')
# test_vector = np.save('toy_data/test_vector', test_vector)
