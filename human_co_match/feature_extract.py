# -*- coding: utf-8 -*-
"""
 Time : 2021/5/15 8:39
 Author : huangkai
 File : feature_extract.py
 Software: PyCharm 
 mail:18707125049@163.com
 
"""
import pandas as pd


# 求职者
# 求职者
def person_feature(path):
    person = pd.read_csv(path + "person.csv", header=0,
                         names=["求职者编号", "性別", "工作年限", "最高学历", "应聘者专业", "年龄", "最近工作岗位", "最近所在行业", "当前工作所在地", "语言能力",
                                "专业特长"], encoding='gb18030')
    person_sex = (person.性別 == "女").astype("float")
    person_edu = person.最高学历.map(
        {"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
    person_major = person.应聘者专业.astype("category")
    return person


# 求职意向
def person_intent(path):
    intent_table = pd.read_csv(path + "person_cv.csv", header=0,
                               names=["求职者编号", "自荐信", "岗位类别", "工作地点", "所在行业", "可到职天数", "其他说明"])
    intent_table["自荐信字数"] = intent_table.自荐信.str.len()
    return intent_table


# 工作经历

def work_history(path):
    work_history = pd.read_csv(path + "person_job_hist.csv", header=0,
                               names=["求职者编号", "岗位类别", "单位所在地", "单位所属行业", "主要业绩"])
    work_history["主要业绩字数"] = work_history.主要业绩.str.len()
    work_history_table = work_history.groupby("求职者编号").aggregate(
        {"岗位类别": "count", "主要业绩字数": ["mean", "sum"]}).reset_index()
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
    candidate_feature = data.groupby("求职者编号").aggregate({"标签": "mean"}).reset_index()
    candidate_feature.columns = ["求职者编号", "求职者平均被匹配数"]
    job_feature = data.groupby("岗位编号").aggregate({"标签": "mean"}).reset_index()
    job_feature.columns = ["岗位编号", "岗位平均被匹配数"]
    return candidate_feature, job_feature


def get_k_fold_data(train_data,
                    person,
                    intent_table,
                    job_table,
                    project_history_table,
                    work_history_table,
                    candidate_data,
                    job_data):
    k = 5
    train = None
    for i in range(k):
        data_label = train_data[train_data.index % k == i].reset_index(drop=True)
        data_feature = train_data[train_data.index % k != i].reset_index(drop=True)

        data_table = get_data(data_label,
                              data_feature,
                              person,
                              intent_table,
                              job_table,
                              project_history_table,
                              work_history_table,
                              candidate_data,
                              job_data)
        train = pd.concat([train, data_table], ignore_index=True)
    return train


def get_data(data,
             feature,
             person,
             intent_table,
             job_table,
             project_history_table,
             work_history_table,
             candidate_data,
             job_data
             ):
    candidate_feature, job_feature = match_current(feature)

    data = data.merge(person, on="求职者编号", how="left")
    data = data.merge(intent_table, on="求职者编号", how="left")
    data = data.merge(job_table, on="岗位编号", how="left")
    data = data.merge(project_history_table, on="求职者编号", how="left")
    data = data.merge(work_history_table, on="求职者编号", how="left")
    data = data.merge(candidate_data, on="求职者编号", how="left")
    data = data.merge(job_data, on="岗位编号", how="left")
    data = data.merge(candidate_feature, on="求职者编号", how="left")
    data = data.merge(job_feature, on="岗位编号", how="left")
    data["工作地点符合否"] = (data.工作地点 == data.岗位工作地点).astype("float")
    #     print(data.columns)
    data_feature = data[["岗位编号", "求职者编号", "标签", "性別", "工作年限", "最高学历", "应聘者专业", "年龄", "自荐信字数", "可到职天数"
        , "项目经验数", "工作经历数", "平均主要业绩字数", "总主要业绩字数", "招聘对象代码", "招聘对象", "岗位最低学历", "岗位工作年限", "具体要求字数", "工作地点符合否"
        , "求职者求职数", "岗位求职者数", "求职者平均被匹配数", "岗位平均被匹配数",
                         ]]

    data_feature = data_feature[["岗位编号", "求职者编号", "标签"] + [column for column in data_feature.columns if
                                                           column not in ["岗位编号", "求职者编号", "标签"]]]

    return data_feature


def feature_process(temp_data):
    data_all_feature = pd.get_dummies(temp_data[['性別', '工作年限', '最高学历', '应聘者专业', '可到职天数', '招聘对象代码', '招聘对象',
                                                 '岗位最低学历', '工作经历数', '岗位工作年限', '工作地点符合否', "求职者平均被匹配数", "岗位平均被匹配数",
                                                 "求职者求职数", "岗位求职者数"]])  # [['性别']]#[']性别']]
    data_all_feature['年龄'] = temp_data['年龄']
    data_all_feature['项目经验数'] = temp_data['项目经验数']
    data_all_feature['具体要求字数'] = temp_data['具体要求字数']
    data_all_feature['自荐信字数'] = temp_data['自荐信字数']
    data_all_feature['平均主要业绩字数'] = temp_data['平均主要业绩字数']
    data_all_feature['总主要业绩字数'] = temp_data['总主要业绩字数']
    data_all_feature['岗位编号'] = temp_data['岗位编号']
    data_all_feature['求职者编号'] = temp_data['求职者编号']
    data_all_feature['标签'] = temp_data['标签']
    return data_all_feature
