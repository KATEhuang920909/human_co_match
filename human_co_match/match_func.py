# -*- coding: utf-8 -*-
"""
 Time : 2021/5/19 2:52
 Author : huangkai
 File : match_func.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
"""2021-05-19 02:52:20 文本特征包括各类相似度特征，关键词匹配特征，文本的embedding特征，也可以考虑纯文本的匹配"""
import numpy as np
from simhash import Simhash
import difflib


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
def sim_jaccard(sentence1: str, sentence2: str) -> float:  # terms_reference为源句子，terms_model为候选句子
    """
    这里用到jaccard匹配
    :param text1: 投递者专业
    :param text2: 岗位要求专业
    :return: 匹配值
    """
    sentence1 = str(sentence1)
    sentence2 = str(sentence2)
    if sentence1 and sentence2:
        temp = 0
        for i in sentence1:
            if i in sentence2:
                temp = temp + 1
        fenmu = len(sentence2) + len(sentence1) - temp  # 并集
        jaccard_coefficient = float(temp / fenmu)  # 交集
        return jaccard_coefficient


def sim_edit(sentence1: str, sentence2: str) -> float:
    """编辑距离归一化后计算相似度"""
    sentence1 = str(sentence1)
    sentence2 = str(sentence2)
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

    if sentence2 and sentence1:
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
    if sentence1 and sentence2:
        sentence1=str(sentence1)
        sentence2=str(sentence2)
        a_simhash = Simhash(sentence1)
        b_simhash = Simhash(sentence2)
        max_hashbit = max(len(bin(a_simhash.value)), len(bin(b_simhash.value)))
        # 汉明距离
        distince = a_simhash.distance(b_simhash)
        similar = 1 - distince / max_hashbit
        return similar


# 判断相似度的方法，用到了difflib库
def difflibs(str1: str, str2: str) -> float:
    str1 = str(str1)
    str2 = str(str2)
    return difflib.SequenceMatcher(a=str1, b=str2).quick_ratio()


# def gat_match_function(person_feature,
#                        person_intent,
#                        cert_table,
#                        work_history,
#                        certificate,
#                        project_history,
#                        job_table):
#     """
#
#     :param person_feature: 求职者信息
#     :param person_intent: 求职者意向
#     :param work_history: 求职者历史工作信息
#     :param certificate: 求职者证书
#     :param project_history: 求职者项目经历
#     :param job_table: 岗位需求
#     :return: 匹配特征值
#     """
#     person_feature_text = person_feature[["应聘者专业", "最近工作岗位", "最近所在行业", "专业特长"]]
#     person_intent_text = person_intent[["自荐信", "岗位类别", "所在行业", ]]
#     cert_table_text  = cert_table[[]]


# 执行方法进行验证
if __name__ == '__main__':
    str1 = '会计学'
    b = '行政事务'
    print(difflibs(str1, b))

    print(simhash(str1, b))
    print(sim_edit(str1, b))
    # print(cosine_similarity("工程管理", "工商管理"))
    print(sim_jaccard(str1, b))
