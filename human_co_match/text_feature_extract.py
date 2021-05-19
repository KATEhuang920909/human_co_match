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
import difflib
import Levenshtein

# 文本匹配方法


# 语义向量相似度
def cosine_similarity(vec1, vec2) -> float:
    """
    """

    num = vec1.dot(vec2.T)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos

    return sim


##############字符串相似度4种#########################

# edit distance
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


def sim_edit(sentence1: str, sentence2: str) -> float:
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
    res = edit_distance(sentence1, sentence2)
    # 2. 归一化到0~1
    maxLen = max(len(sentence1), len(sentence2))
    sim = 1 - res * 1.0 / maxLen
    return sim


def simhash(sentence1: str, sentence2: str) -> float:
    """
    求两文本的相似度
    :param text_a:
    :param text_b:
    :return:
    """
    a_simhash = Simhash(sentence1)
    b_simhash = Simhash(sentence2)
    max_hashbit = max(len(bin(a_simhash.value)), len(bin(b_simhash.value)))
    # 汉明距离
    distince = a_simhash.distance(b_simhash)
    similar = 1 - distince / max_hashbit
    return similar


# 判断相似度的方法，用到了difflib库
def get_equal_rate_1(str1, str2):
    return difflib.SequenceMatcher(a=str1, b=str2).quick_ratio()


# 执行方法进行验证
if __name__ == '__main__':
    str1 = '任正非称，对华为不会出现“断供”这种极端情况，我们已经做好准备了。任正非称，今年春节时，我们判断出现这种情况是2年以后。\
   我还有两年时间去足够足够准备了。孟晚舟事件时我们认为这个时间提前了，我们春节都在加班。保安、清洁工、服务人员，春节期间有5000人\
   都在加班，加倍工资都在供应我们的战士战斗，大家都在抢时间。（新浪科技）'
    b = ' 任正非称，对华为不会出现“断供”这种极端情况，我们已经做好准备了。任正非称，今年春节时，我们判断出现这种情况是2年以后。\
   我还有两年时间去足够足够准备了。孟晚舟事件时我们认为这个时间提前了，我们春节都在加班。保安、清洁工、服务人员，春节期间有5000人\
   都在加班，加倍工资都在供应我们的战士战斗，大家都在抢时间。'
    print(get_equal_rate_1(str1, b))

    print(simhash(str1, b))
    print(sim_edit(str1, b))
    # print(cosine_similarity("工程管理", "工商管理"))
    print(sim_jaccard(str1, b))
