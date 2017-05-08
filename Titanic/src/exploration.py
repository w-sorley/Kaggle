#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd


import matplotlib.pyplot as plt

data_train = pd.read_csv("../data/train.csv")
# print(data_train.info())
# print(data_train.describe())
fig = plt.figure()
fig.set(alpha=0.2)


plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"获救情况（1为获救）")
plt.ylabel(u"人数")

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布（１为获救）")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级乘客年龄分布")
plt.legend((u'头等舱', u'二等舱', u'三等舱'), loc='best')

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")

# fig = plt.figure()
# fig.set(alpha=0.2)

pclassSurvived = data_train.Pclass[data_train.Survived == 1].value_counts()
pclassDead = data_train.Pclass[data_train.Survived == 0].value_counts()
df = pd.DataFrame({u"获救": pclassSurvived, u"未获救": pclassDead})
df.plot(kind='bar', stacked=True)
plt.title("各等级乘客获救情况")
plt.xlabel("乘客等级")
plt.ylabel("人数")

survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({'男性': survived_m, '女性': survived_f})
df.plot(kind='bar', stacked=True)
plt.title("获救者性别分布情况")
plt.xlabel("获救")
plt.ylabel("人数")


fig = plt.figure()
fig.set(alpha=0.88)
plt.title("根据船舱等级和性别观察获救情况")
ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == "female"][data_train.Pclass != 3].value_counts().plot(
    kind='bar', label="female highclass", color="#FA2479")
ax1.set_xticklabels(["获救", "未获救"], rotation=0)
ax1.legend(["女性／高等仓"], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == "female"][data_train.Pclass == 3].value_counts().plot(
    kind='bar', label="female lowclass", color="pink")
ax2.set_xticklabels(["未获救", "获救"], rotation=0)
ax2.legend(["女性／低等仓"], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == "male"][data_train.Pclass != 3].value_counts().plot(
    kind='bar', label="male highclass", color="blue")
ax3.set_xticklabels(["未获救", "获救"], rotation=0)
ax3.legend(["男性／高等仓"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == "male"][data_train.Pclass == 3].value_counts().plot(
    kind='bar', label="male lowclass", color="green")
ax4.set_xticklabels(["未获救", "获救"], rotation=0)
ax4.legend(["男性／低等仓"], loc='best')

embarkedSurvived = data_train.Embarked[data_train.Survived == 1].value_counts()
embarkedDead = data_train.Embarked[data_train.Survived == 0].value_counts()
df = pd.DataFrame({'获救': embarkedSurvived, '未获救': embarkedDead})
df.plot(kind='bar', stacked='true')
plt.title('各登录港口乘客获救情况')
plt.xlabel('登录港口')
plt.ylabel('人数')

plt.show()
