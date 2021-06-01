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
import pandas as pd
from text_matching.bert.basic_feature_extract import bert_encoder
jieba.load_userdict(r"D:\learning\competition\人岗匹配\human_co_match\human_co_match\data\dict.txt")


def text_clean(txt):
    result = re.split(r"[{}]".format(eg_punc + ch_punc), txt)
    return ''.join(result)


def text_tokenizer(txt):
    sentence = text_clean(txt)
    sentence = jieba.lcut(sentence)
    return sentence





# 投递者文本拼接，岗位文本拼接
def text_concat(path, data):  # 总共六张表
    person = pd.read_csv(path + "person.csv", header=0,
                         names=["求职者编号", "性别", "工作年限", "最高学历", "应聘者专业", "年龄", "最近工作岗位", "最近所在行业", "当前工作所在地", "语言能力",
                                "专业特长"], encoding="gb18030")
    intent_table = pd.read_csv(path + "person_cv.csv", header=0,
                               names=["求职者编号", "自荐信", "求职意向岗位类别", "求职意向工作地点", "求职意向所在行业", "可到职天数", "其他说明"])
    work_history = pd.read_csv(path + "person_job_hist.csv", header=0,
                               names=["求职者编号", "工作经历岗位类别", "工作经历单位所在地", "工作经历单位所属行业", "工作经历主要业绩"])
    cert_table = pd.read_csv(path + "person_pro_cert.csv", header=0, names=["求职者编号", "专业证书名称", "备注"])
    project_history = pd.read_csv(path + "person_project.csv", header=0,
                                  names=["求职者编号", "项目名称", "项目说明", "职责说明", "关键技术"])
    job_table = pd.read_csv(path + "recruit.csv", header=0,
                            names=["岗位编号", "招聘对象代码", "招聘对象", "招聘职位", "对应聘者的专业要求", "岗位最低学历", "岗位工作地点", "岗位工作年限", "具体要求"])
    print("开始处理")
    person["求职者文本内容"], intent_table["投递意向文本内容"], work_history["工作经历文本内容"] = [""] * person.shape[0], \
                                                                            [""] * intent_table.shape[0], \
                                                                            [""] * work_history.shape[0]
    cert_table["证书文本内容"], project_history["项目经历文本内容"], job_table["岗位文本内容"] = [""] * cert_table.shape[0], \
                                                                             [""] * project_history.shape[0], \
                                                                             [""] * job_table.shape[0],
    for col in person.columns[1:-1]:
        person["求职者文本内容"] += person[col].astype(str)
        person["求职者文本内容"] += '0v0'
    print("求职者文本内容处理完毕")
    for col in intent_table.columns[1:-1]:
        intent_table["投递意向文本内容"] += intent_table[col].astype(str)
        intent_table["投递意向文本内容"] += '0v0'
    print("投递意向文本内容处理完毕")
    for col in work_history.columns[1:-1]:
        work_history["工作经历文本内容"] += work_history[col].astype(str)
        work_history["工作经历文本内容"] += '0v0'
    print("工作经历文本内容处理完毕")
    work_history = work_history.groupby('求职者编号')['工作经历文本内容'].apply(lambda x: x.str.cat(sep='///')).reset_index()
    for col in cert_table.columns[1:-1]:
        cert_table["证书文本内容"] += cert_table[col].astype(str)
        cert_table["证书文本内容"] += '0v0'
    cert_table = cert_table.groupby('求职者编号')['证书文本内容'].apply(lambda x: x.str.cat(sep='///')).reset_index()
    print("证书文本内容处理完毕")
    for col in project_history.columns[1:-1]:
        project_history["项目经历文本内容"] += project_history[col].astype(str)
        project_history["项目经历文本内容"] += '0v0'
    print("项目经历文本内容处理完毕")
    project_history = project_history.groupby('求职者编号')['项目经历文本内容'].apply(lambda x: x.str.cat(sep='///')).reset_index()
    for col in job_table.columns[1:-1]:
        job_table["岗位文本内容"] += job_table[col].astype(str)
        job_table["岗位文本内容"] += '0v0'
    print("岗位文本内容处理完毕")
    data = pd.merge(data, person[["求职者编号", "求职者文本内容"]], on="求职者编号", how='left')
    print("1", data.shape)
    data = pd.merge(data, intent_table[["求职者编号", "投递意向文本内容"]], on="求职者编号", how='left')
    print("2", data.shape)
    data = pd.merge(data, work_history[["求职者编号", "工作经历文本内容"]], on="求职者编号", how='left')
    print("3", data.shape)
    data = pd.merge(data, cert_table[["求职者编号", "证书文本内容"]], on="求职者编号", how='left')
    print("4", data.shape)
    data = pd.merge(data, project_history[["求职者编号", "项目经历文本内容"]], on="求职者编号", how='left')
    print("5", data.shape)
    data = pd.merge(data, job_table[["岗位编号", "岗位文本内容"]], on="岗位编号", how='left')
    print("6", data.shape)
    print("处理完毕")
    return data


if __name__ == '__main__':
    # print(text_tokenizer("/普通话二级甲，普通话二甲，普通话二级乙，普通话二乙，"))
    print(text_clean("【园艺学】"))
