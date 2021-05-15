# python 3.9.2
# python包 lightgbm 3.1.1
# python包 numpy 1.20.1
# python包 pandas 1.2.2
# -*-coding:utf-8 -*-
# input：
#	trainset/person.csv
#	trainset/person_cv.csv
#	trainset/person_job_hist.csv
#	trainset/person_pro_cert.csv
#	trainset/person_project.csv
#	trainset/recruit.csv
#	trainset/recruit_folder.csv
#	testset/recruit_folder.csv
#
# output：
# 	result.csv
#
# 0.8516
#
import numpy
import pandas
import random
import sklearn
import lightgbm

# 求职者
person = pandas.read_csv("trainset/person.csv", header=0,
                         names=["求职者编号", "性別", "工作年限", "最高学历", "应聘者专业", "年龄", "最近工作岗位", "最近所在行业", "当前工作所在地", "语言能力",
                                "专业特长"], encoding='gb18030')
person_sex = (person.性別 == "女").astype("float")
person_edu = person.最高学历.map({"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
person_major = person.应聘者专业.astype("category")

# 求职意向
intent_table = pandas.read_csv("trainset/person_cv.csv", header=0,
                               names=["求职者编号", "自荐信", "岗位类别", "工作地点", "所在行业", "可到职天数", "其他说明"])
intent_table["自荐信字数"] = intent_table.自荐信.str.len()

# 工作经历
work_history = pandas.read_csv("trainset/person_job_hist.csv", header=0,
                               names=["求职者编号", "岗位类别", "单位所在地", "单位所属行业", "主要业绩"])
work_history["主要业绩字数"] = work_history.主要业绩.str.len()

# 证书
cert_table = pandas.read_csv("trainset/person_pro_cert.csv", header=0, names=["求职者编号", "专业证书名称", "备注"])

# 项目经历
project_history = pandas.read_csv("trainset/person_project.csv", header=0,
                                  names=["求职者编号", "项目名称", "项目说明", "职责说明", "关键技术"])

# 岗位表
job_table = pandas.read_csv("trainset/recruit.csv", header=0,
                            names=["岗位编号", "招聘对象代码", "招聘对象", "招聘职位", "对应聘者的专业要求", "岗位最低学历", "岗位工作地点", "岗位工作年限", "具体要求"])
job_table.招聘对象代码 = job_table.招聘对象代码.fillna(-1).astype("category")
job_table.招聘对象 = job_table.招聘对象.astype("category")
job_table.岗位最低学历 = job_table.岗位最低学历.map(
    {"其它": 0, "中专": 1, "高中（职高、技校）": 2, "大专": 3, "大学本科": 4, "硕士研究生": 5, "博士研究生": 6, "博士后": 7})
job_table.岗位工作年限 = job_table.岗位工作年限.map({"不限": -1, "应届毕业生": 0, "0至1年": 0, "1至2年": 1, "3至5年": 3, "5年以上": 5})
job_table["具体要求字数"] = job_table.具体要求.str.len()

work_history_table = work_history.groupby("求职者编号").aggregate({"岗位类别": "count", "主要业绩字数": ["mean", "sum"]}).reset_index()
work_history_table.columns = ["求职者编号", "工作经历数", "平均主要业绩字数", "总主要业绩字数"]
project_history_table = project_history.groupby("求职者编号").aggregate({"项目名称": "count"}).reset_index()
project_history_table.columns = ["求职者编号", "项目经验数"]

train_data = pandas.read_csv("trainset/recruit_folder.csv", header=0, names=["岗位编号", "求职者编号", "标签"])
test_data = pandas.read_csv("testset/recruit_folder.csv", header=0, names=["岗位编号", "求职者编号", "标签"])

data = pandas.concat([test_data, train_data], ignore_index=True)
candidate_data = data.groupby("求职者编号").aggregate({"岗位编号": "count"}).reset_index()
candidate_data.columns = ["求职者编号", "求职者数"]
job_data = data.groupby("岗位编号").aggregate({"求职者编号": "count"}).reset_index()
job_data.columns = ["岗位编号", "岗位数"]


def get_data(data, feature):
    candidate_feature = feature.groupby("求职者编号").aggregate({"标签": "mean"}).reset_index()
    candidate_feature.columns = ["求职者编号", "求职者平均标签"]
    job_feature = feature.groupby("岗位编号").aggregate({"标签": "mean"}).reset_index()
    job_feature.columns = ["岗位编号", "岗位平均标签"]

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

    data_feature = data.loc[["岗位编号", "求职者编号", "标签", "性別", "工作年限", "最高学历", "应聘者专业", "年龄", "自荐信字数", "可到职天数"
        , "项目经验数", "工作经历数", "平均主要业绩字数", "总主要业绩字数", "招聘对象代码", "招聘对象", "岗位最低学历", "岗位工作年限", "具体要求字数", "工作地点符合否"
        , "求职者数", "岗位数", "求职者平均标签", "岗位平均标签"
                             ]]

    data_feature = data_feature.loc[["岗位编号", "求职者编号", "标签"] + [column for column in data_feature.columns if
                                                               column not in ["岗位编号", "求职者编号", "标签"]]]

    return data_feature


k = 4
train = None
for i in range(k):
    train_label = train_data[train_data.index % k == i].reset_index(drop=True)
    train_feature = train_data[train_data.index % k != i].reset_index(drop=True)

    data_table = get_data(train_label, train_feature)
    train = pandas.concat([train, data_table], ignore_index=True)

model = lightgbm.train(train_set=lightgbm.Dataset(train.iloc[:, 3:], label=train.标签)
                       , num_boost_round=500,
                       params={"objective": "binary", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 32,
                               "verbose": -1, "bagging_fraction": 0.8, "feature_fraction": 0.8})
test_table = get_data(test_data, train_data)
test = test_table.loc[:, ["岗位编号", "求职者编号"]]
test["预测打分"] = model.predict(test_table.iloc[:, 3:])
test = test.sort_values("预测打分", ascending=False, ignore_index=True)
test["预测"] = 0
test.loc[:int(0.15 * len(test)), ["预测"]] = 1

submit = test.loc[:, ["岗位编号", "求职者编号", "预测"]]
submit.columns = ["RECRUIT_ID", "PERSON_ID", "LABEL"]
submit.to_csv("result.csv", index=False)
