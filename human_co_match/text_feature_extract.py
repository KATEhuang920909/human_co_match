# -*- coding: utf-8 -*-
"""
 Time : 2021/5/19 2:52
 Author : huangkai
 File : text_feature_extract.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
"""2021-05-19 02:52:20 文本特征包括各类相似度特征，关键词匹配特征，文本的embedding特征，也可以考虑纯文本的匹配"""
import numpy as np
from simhash import Simhash







# 文本匹配方法
# jaccard
def sim_jaccard(grams_reference, grams_model):  # terms_reference为源句子，terms_model为候选句子
    """
    这里用到jaccard匹配
    :param text1: 投递者专业
    :param text2: 岗位要求专业
    :return: 匹配值
    """
    if grams_reference:
        temp = 0
        for i in grams_reference:
            if i in grams_model:
                temp = temp + 1
        fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
        jaccard_coefficient = float(temp / fenmu)  # 交集
        return jaccard_coefficient
    else:
        return 1  # 不限专业则符合要求


# 余弦距离
def cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    compute normalized COSINE similarity.
    :param sentence1: English sentence.
    :param sentence2: English sentence.
    :return: normalized similarity of two input sentences.
    """
    seg1 = sentence1.strip(" ").split(" ")
    seg2 = sentence2.strip(" ").split(" ")
    word_list = list(set([word for word in seg1 + seg2]))
    word_count_vec_1 = []
    word_count_vec_2 = []
    for word in word_list:
        word_count_vec_1.append(seg1.count(word))
        word_count_vec_2.append(seg2.count(word))

    vec_1 = np.array(word_count_vec_1)
    vec_2 = np.array(word_count_vec_2)

    num = vec_1.dot(vec_2.T)
    denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos

    return sim


# edit distance

def sim_edit(s1, s2):
    """编辑距离归一化后计算相似度"""

    # def edit_sim(s1, s2):
    #     import Levenshtein  # 第三方库实现
    #     maxLen = max(len(s1), len(s2))
    #     dis = Levenshtein.distance(s1, s2)
    #     sim = 1 - dis * 1.0 / maxLen
    #     return sim
    def edit_distance(str1, str2):
        len1, len2 = len(str1), len(str2)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                temp = 0 if str1[i - 1] == str2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + temp, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        return dp[len1][len2]

    # 1. 计算编辑距离
    res = edit_distance(s1, s2)
    # 2. 归一化到0~1
    maxLen = max(len(s1), len(s2))
    sim = 1 - res * 1.0 / maxLen
    return sim




def simhash(text_a, text_b):
    """
    求两文本的相似度
    :param text_a:
    :param text_b:
    :return:
    """
    a_simhash = Simhash(text_a)
    b_simhash = Simhash(text_b)
    max_hashbit = max(len(bin(a_simhash.value)), len(bin(b_simhash.value)))
    # 汉明距离
    distince = a_simhash.distance(b_simhash)
    similar = 1 - distince / max_hashbit
    return similar


if __name__ == '__main__':
    print(simhash("工程管理", "工商管理"))
    print(sim_edit("工程管理", "工商管理"))
    print(cosine_similarity("工程管理", "工商管理"))
    print(sim_jaccard("工程管理", "工商管理"))
