# -*- coding: utf-8 -*-
"""
 Time : 2021/5/15 8:39
 Author : huangkai
 File : feature_extract.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""

"""2021-05-16 01:41:05 添加最近工作岗位、最近所在行业、当前工作所在地三个转离散指标"""
"""2021-05-16 01:57:53工作年限是否符合、专业是否符合（语言模型判断相似度）"""
"""2021-05-16 02:30:00语言能力：普通话标准、流利、精通、二级甲等（二甲、二级甲）、二级乙等（二级乙、二乙）、粤语、四级、六级、八级、日语"""
"""2021-05-19 02:48:16  1.证书的挖掘；2.单位所属行业;3.单位所在地的挖掘"""
"""2021-05-23 00:36:06  添加特征：是否有专业限制"""
import pandas as pd
from data_preprocess import text_clean, text_tokenizer
import re
import numpy as np
from match_func import simhash, sim_jaccard, sim_edit, difflibs
from tqdm import tqdm
import time as t


# 求职者
# 求职者
def person_feature(path):
    person = pd.read_csv(path + "person.csv", header=0,
                         names=["求职者编号", "性别", "工作年限", "最高学历", "应聘者专业", "年龄", "最近工作岗位", "最近所在行业", "当前工作所在地", "语言能力",
                                "专业特长"], encoding="gb18030")
    person_sex = (person.性别 == "女").astype("float")
    person.最高学历 = person.最高学历.map(
        {"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
    person.性别 = person.性别.map({"女": 0, "男": 1})
    person.应聘者专业 = person.应聘者专业.astype("category")
    person.应聘者专业 = person.应聘者专业.apply(lambda x: text_clean(x))
    return person


# 求职意向
def person_intent(path):
    intent_table = pd.read_csv(path + "person_cv.csv", header=0,
                               names=["求职者编号", "自荐信", "求职意向岗位类别", "求职意向工作地点", "求职意向所在行业", "可到职天数", "其他说明"])
    intent_table["自荐信字数"] = intent_table.自荐信.str.len()
    intent_table["求职意向岗位类别"] = intent_table.求职意向岗位类别.astype("category")
    intent_table["求职意向岗位类别"] = intent_table.求职意向岗位类别.apply(lambda x: text_clean(x))
    return intent_table


# 工作经历

def work_history(path):
    work_history = pd.read_csv(path + "person_job_hist.csv", header=0,
                               names=["求职者编号", "工作经历岗位类别", "工作经历单位所在地", "工作经历单位所属行业", "工作经历主要业绩"])
    work_history["主要业绩字数"] = work_history.工作经历主要业绩.str.len()
    work_history_table = work_history.groupby("求职者编号").aggregate(
        {"工作经历岗位类别": "count", "主要业绩字数": ["mean", "sum"]}).reset_index()
    work_history_table.columns = ["求职者编号", "工作经历数", "平均主要业绩字数", "总主要业绩字数"]
    return work_history, work_history_table


def certificate(path):
    # 证书
    cert_table = pd.read_csv(path + "person_pro_cert.csv", header=0, names=["求职者编号", "专业证书名称", "备注"])
    return cert_table


# 项目经历
def project_history(path):
    project_history = pd.read_csv(path + "person_project.csv", header=0,
                                  names=["求职者编号", "项目名称", "项目说明", "职责说明", "关键技术"])
    project_history_table = project_history.groupby("求职者编号").aggregate({"项目名称": "count"}).reset_index()
    project_history_table.columns = ["求职者编号", "项目经验数"]
    return project_history, project_history_table


# 岗位表
def job_table(path):
    job_table = pd.read_csv(path + "recruit.csv", header=0,
                            names=["岗位编号", "招聘对象代码", "招聘对象", "招聘职位", "对应聘者的专业要求", "岗位最低学历", "岗位工作地点", "岗位工作年限", "具体要求"])
    job_table.招聘对象代码 = job_table.招聘对象代码.fillna(-1).astype("category")
    job_table.招聘对象 = job_table.招聘对象.astype("category")
    job_table.岗位最低学历 = job_table.岗位最低学历.map(
        {"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
    job_table.岗位工作年限 = job_table.岗位工作年限.map({"不限": -1, "应届毕业生": 0, "0至1年": 0, "1至2年": 1, "3至5年": 3, "5年以上": 5})
    job_table["具体要求字数"] = job_table.具体要求.str.len()
    job_table["是否有专业限制"] = job_table.对应聘者的专业要求.apply(lambda x: False if pd.isnull(x) else True)  # 有：true 无：false
    return job_table


# 求职者投递情况和岗位的投递情况
def diliver_current(data_all):
    candidate_data = data_all.groupby("求职者编号").aggregate({"岗位编号": "count"}).reset_index()
    candidate_data.columns = ["求职者编号", "求职者求职数"]
    job_data = data_all.groupby("岗位编号").aggregate({"求职者编号": "count"}).reset_index()
    job_data.columns = ["岗位编号", "岗位求职者数"]
    return candidate_data, job_data


# 岗位的匹配情况和求职者的匹配情况
def match_current(data):
    result = []
    ls = ["岗位编号", "求职者编号", "性别", "工作年限", "最高学历", "应聘者专业", "年龄", "可到职天数", "项目经验数", "工作经历数",
          "工作地点符合否", "工作年限是否符合", "岗位工作地点", "求职意向工作地点", "求职意向岗位类别"]
    for label in ls:
        temp_data = data.groupby(label).aggregate({"标签": "mean"}).reset_index()
        temp_data.columns = [label, f"{label}平均被匹配数"]
        result.append(temp_data)

    return result


# 岗位工作年限是否符合
def work_year_match(data):
    # {"不限": -1, "应届毕业生": 0, "0至1年": 0, "1至2年": 1, "3至5年": 3, "5年以上": 5}
    if data.岗位工作年限 == -1:
        return 1
    if data.岗位工作年限 in [0, 1]:
        if data.工作年限 in [0, 1, 2]:
            return 1
    if data.岗位工作年限 == 3:
        if data.工作年限 in [3, 4, 5]:
            return 1
    if data.岗位工作年限 == 5:
        if data.工作年限 >= 5:
            return 1
    return 0


# 英语语言能力特征离散
def english_language_feature(txt):
    if txt:
        txt = str(txt)
    else:
        return "UNKNOWN"
    if ("雅思" in txt) or ("IELTS" in txt) or ("托福" in txt) or ("TOEFL" in txt):
        return "托福雅思"
    elif ("八级" in txt) or ("8级" in txt):
        return "八级"
    elif ("6级" in txt) or ("6级" in txt) or ("CET-6" in txt) or ("CET6" in txt):
        return "六级"
    elif ("四级" in txt) or ("4级" in txt) or ("CET-4" in txt) or ("CET4" in txt):
        return "四级"
    return "UNKNOWN"


# 汉语语言能力
def chinese_language_feature(txt):
    if txt:
        txt = str(txt)
    else:
        return "UNKNOWN"
    if ("二级甲" in txt) or ("二甲" in txt) or ("2级甲" in txt) or ("2甲" in txt):
        return "二级甲"
    if ("二级乙" in txt) or ("二乙" in txt) or ("2级乙" in txt) or ("2乙" in txt):
        return "二级乙"
    language = ["普通话", "汉"]
    degree = ["标准", "流利", "精通", "流畅", "熟练", "一般"]
    for l in language:
        for d in degree:
            if l in txt:
                if d in txt:
                    if l == "普通话":
                        if abs(txt.index(l) - txt.index(d)) <= 5:
                            return l + d
                    if l == "汉":
                        if abs(txt.index(l) - txt.index(d)) <= 5:
                            return "普通话" + d
    return "UNKNOWN"


# 粤语语言能力
def guangdong_language_feature(txt):
    if txt:
        txt = str(txt)
    else:
        return "UNKNOWN"
    language = ["广东话", "粤语"]
    degree = ["标准", "流利", "精通", "流畅", "熟练", "一般"]
    for l in language:
        for d in degree:
            if l in txt:
                if d in txt:
                    if l == "粤语":
                        if abs(txt.index(l) - txt.index(d)) <= 5:
                            return "广东话" + d
                        else:
                            return "广东话流利"
                else:
                    if (l == "广东话") or (l == "粤语"):
                        return "广东话流利"
    return "UNKNOWN"


# 日语语言能力
def japanese_language_feature(txt):
    if txt:
        txt = str(txt)
    else:
        return "UNKNOWN"
    language = "日语"
    if language in txt:
        return "日语"
    return "UNKNOWN"


# 投递岗位的名词中是否在工作业绩描述中出现
def job_name_work_history_match(name, txt):
    """
    :todo:
    :param name:岗位名称
    :param txt:业绩描述
    :return:
    """
    pass


# 男性占比
def male_rate(data):
    s = data.sum()  # 男性个数
    t = data.count()  # 总人数
    return s / t  # 男性个数占比


# 众数
def mode(data):
    data = list(data)
    res = []
    for num in data:
        if not pd.isnull(num) and int(num) >= 0:
            res.append(int(num))

    counts = np.bincount(res)
    # 返回众数
    if counts.shape[0] == 0:
        return None
    return np.argmax(counts)


# 众数占比
def mode_rate(data):
    data = list(data)
    res = []
    for num in data:
        if not pd.isnull(num) and int(num) >= 0:
            res.append(int(num))
    counts = np.bincount(res)
    if counts.shape[0] == 0:
        return None
    # 返回众数占比
    return counts[np.argmax(counts)] / sum(counts)


# 招聘岗位的投递记录中：求职意向岗位，求职者专业，（年龄、工作经历数、项目经验数、可到职天数）统计指标、工作地点符合占比、
# 统计方式：train\test联合统计
def deliver_record_feature(data_feature):
    for f1, f2 in tqdm([["岗位编号", "岗位类别"], ["岗位编号", "应聘者专业"], ["岗位编号", "工作地点"]]):
        data_feature[f"{f1}_{f2}_nunique"] = data_feature.groupby(
            [f1])[f2].transform("nunique")
    # 最大值、最小值、均值、标准差、众数、众数占比
    for f1, f2 in tqdm([["岗位编号", "年龄"], ["岗位编号", "工作年限"], ["岗位编号", "工作经历数"], ["岗位编号", "项目经验数"], ["岗位编号", "总主要业绩字数"],
                        ["岗位编号", "最高学历"], ["岗位编号", "可到职天数"]]):
        df_temp1 = data_feature.groupby(f1)[f2].agg(
            [(f"{f1}_{f2}_mean", "mean"),
             (f"{f1}_{f2}_max", "max"),
             (f"{f1}_{f2}_min", "min"),
             (f"{f1}_{f2}_std", "std"),
             (f"{f1}_{f2}_median", np.median),
             (f"{f1}_{f2}_mode", mode),
             (f"{f1}_{f2}_mode_rate", mode_rate)]
        ).reset_index()
        data_feature = data_feature.merge(df_temp1, how="left")

    # 男性占比、女性个数占比、工作地点符合否占比。
    for f1, f2 in [["岗位编号", "性别"], ["岗位编号", "工作地点符合否"]]:  # 性别众数及占比，工作地点符合及占比
        df_temp2 = data_feature.groupby(f1)[f2].agg(
            [
                (f"{f1}_{f2}_mode", mode),
                (f"{f1}_{f2}_mode_rate", mode_rate)]
        ).reset_index()
        data_feature = data_feature.merge(df_temp2, how="left")
    return data_feature


def person_record_feature(data_feature):
    """
    # :todo 可以获取求职者的历史工作的统计信息；
    :param data_feature:
    :return:
    """
    """2021-05-18 01:43:45"""


def get_k_fold_data(train_data,
                    person,
                    intent_table,
                    job_table,
                    project_history,
                    project_history_table,
                    work_history,
                    work_history_table,
                    candidate_data,
                    job_data):
    """
        :todo k-折交叉的标签特征有很多信息可以挖掘
    :param train_data: 元数据
    :param person: 个人信息
    :param intent_table: 投岗倾向
    :param job_table: 岗位信息
    :param project_history_table: 投递者项目经历
    :param work_history_table: 投递者工作经历
    :param candidate_data: 求职者求职数
    :param job_data: 岗位求职者数
    :return:
    """
    """2021-05-18 01:44:22"""
    k = 5
    train = None
    train_data = data_merge(train_data,
                            person,
                            intent_table,
                            job_table,
                            project_history,
                            project_history_table,
                            work_history,
                            work_history_table,
                            candidate_data,
                            job_data)
    for i in range(k):
        data_label = train_data[train_data.index % k == i].reset_index(drop=True)
        data_feature = train_data[train_data.index % k != i].reset_index(drop=True)

        data_table = get_data(data_label,
                              data_feature)
        train = pd.concat([train, data_table], ignore_index=True)
    return train


def data_merge(data,
               person,
               intent_table,
               job_table,
               project_history,
               project_history_table,
               work_history,
               work_history_table,
               candidate_data,
               job_data):
    data = data.merge(person, on="求职者编号", how="left")
    data = data.merge(job_table, on="岗位编号", how="left")

    """work_history column:["求职者编号", "工作经历岗位类别", "工作经历单位所在地", "工作经历单位所属行业", "工作经历主要业绩"])"""
    data_temp = data.merge(work_history, on="求职者编号", how="left")

    for func in ["simhash", "sim_jaccard", "difflibs"]:
        for wh in ["工作经历岗位类别", "工作经历单位所属行业", "工作经历主要业绩"]:
            for jd in ["招聘职位", "对应聘者的专业要求", "具体要求"]:
                st = t.time()
                data_temp[f"{wh}_{jd}_{func}"] = data_temp.apply(lambda x: eval(func)(x[wh], x[jd]),
                                                                 axis=1)
                df_temp1 = data_temp.groupby("求职者编号")[f"{wh}_{jd}_{func}"].agg(
                    [(f"{wh}_{jd}_{func}_mean", "mean"),
                     (f"{wh}_{jd}_{func}_max", "max"),
                     (f"{wh}_{jd}_{func}_min", "min"),
                     (f"{wh}_{jd}_{func}_std", "std")]
                ).reset_index()
                data = data.merge(df_temp1, on="求职者编号", how="left")
                print(f"{wh}_{jd}_{func}，共耗时:", t.time() - st)
    """project history column:["项目名称", "项目说明", "职责说明", "关键技术"]"""
    data_temp2 = data.merge(project_history, on="求职者编号", how="left")
    for func in ["simhash", "sim_jaccard", "difflibs"]:
        for ph in ["项目名称", "项目说明", "职责说明", "关键技术"]:
            for jd in ["招聘职位", "对应聘者的专业要求", "具体要求"]:
                st = t.time()
                data_temp2[f"{ph}_{jd}_{func}"] = data_temp2.apply(lambda x: eval(func)(x[ph], x[jd]),
                                                                   axis=1)
                df_temp2 = data_temp2.groupby("求职者编号")[f"{ph}_{jd}_{func}"].agg(
                    [(f"{ph}_{jd}_{func}_mean", "mean"),
                     (f"{ph}_{jd}_{func}_max", "max"),
                     (f"{ph}_{jd}_{func}_min", "min"),
                     (f"{ph}_{jd}_{func}_std", "std")]
                ).reset_index()
                data = data.merge(df_temp2, on="求职者编号", how="left")
                print(f"{ph}_{jd}_{func}，共耗时:", t.time() - st)

    data = data.merge(intent_table, on="求职者编号", how="left")
    data = data.merge(project_history_table, on="求职者编号", how="left")
    data = data.merge(work_history_table, on="求职者编号", how="left")

    data = data.merge(candidate_data, on="求职者编号", how="left")
    data = data.merge(job_data, on="岗位编号", how="left")
    """求职者个人的匹配特征"""

    for func in ["simhash", "sim_jaccard", "difflibs"]:
        for ud in ["应聘者专业", "最近工作岗位", "最近所在行业", "专业特长",
                   "求职意向岗位类别", "求职意向所在行业", "自荐信"]:
            for jd in ["招聘职位", "对应聘者的专业要求", "具体要求"]:
                st = t.time()
                data[f"{ud}_{jd}_{func}"] = data.apply(lambda x: eval(func)(x[ud], x[jd]), axis=1)
                print(f"{ud}_{jd}_{func}，共耗时:", t.time() - st)

    data["工作地点符合否"] = (data.求职意向工作地点 == data.岗位工作地点).astype("float")
    data["工作年限是否符合"] = data.apply(lambda x: work_year_match(x), axis=1)
    data["汉语语言能力"] = data.语言能力.apply(lambda x: chinese_language_feature(x))
    data["英语语言能力"] = data.语言能力.apply(lambda x: english_language_feature(x))
    data["粤语语言能力"] = data.语言能力.apply(lambda x: guangdong_language_feature(x))
    data["日语语言能力"] = data.语言能力.apply(lambda x: japanese_language_feature(x))

    # 对于工作经历和项目经历，获取每个求职者的工作经历与岗位要求的匹配值，然后获取统计指标：

    # data["专业是否符合"] = data.apply(lambda x: major_match(str(x.对应聘者的专业要求), str(x.应聘者专业)), axis=1)

    return data


def get_data(data,
             feature,
             ):
    result = match_current(feature)
    for df in result:
        data = data.merge(df, on=df.columns[0], how="left")
    #     print(data.columns)
    # data_feature = data[["岗位编号", "求职者编号", "标签", "性别", "工作年限", "最高学历", "应聘者专业", "年龄", "自荐信字数", "可到职天数"
    #                         , "项目经验数", "工作经历数", "平均主要业绩字数", "总主要业绩字数", "招聘对象代码", "招聘对象", "岗位最低学历", "岗位工作年限", "具体要求字数",
    #                      "工作地点符合否", "求职者求职数", "岗位求职者数", "求职者平均被匹配数", "岗位平均被匹配数",
    #                      "专业是否符合", "工作年限是否符合"]
    #                     + ["最近工作岗位", "最近所在行业", "当前工作所在地"]
    #                     + ["汉语语言能力", "英语语言能力", "日语语言能力", "粤语语言能力"]
    #                     + ["岗位工作地点", "工作地点", "岗位类别"]]
    #
    data_feature = data[["岗位编号", "求职者编号", "标签"] + [column for column in data.columns if
                                                   column not in ["岗位编号", "求职者编号", "标签"]]]

    return data_feature


def one_hot_process(temp_data):
    data_all_feature = pd.get_dummies(temp_data[["性别", "工作年限", "最高学历", "应聘者专业", "可到职天数", "招聘对象代码", "招聘对象",
                                                 "岗位最低学历", "工作经历数", "岗位工作年限", "工作地点符合否", "求职者平均被匹配数", "岗位平均被匹配数",
                                                 "求职者求职数", "岗位求职者数",
                                                 "工作年限是否符合"]
                                                + ["最近工作岗位", "最近所在行业", "当前工作所在地"]
                                                + ["汉语语言能力", "英语语言能力", "日语语言能力", "粤语语言能力"]])
    data_all_feature["专业是否符合"] = temp_data["专业是否符合"]
    data_all_feature["年龄"] = temp_data["年龄"]
    data_all_feature["项目经验数"] = temp_data["项目经验数"]
    data_all_feature["具体要求字数"] = temp_data["具体要求字数"]
    data_all_feature["自荐信字数"] = temp_data["自荐信字数"]
    data_all_feature["平均主要业绩字数"] = temp_data["平均主要业绩字数"]
    data_all_feature["总主要业绩字数"] = temp_data["总主要业绩字数"]
    data_all_feature["岗位编号"] = temp_data["岗位编号"]
    data_all_feature["求职者编号"] = temp_data["求职者编号"]
    data_all_feature["标签"] = temp_data["标签"]

    return data_all_feature
