
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


columns = ["var","skewness","curtosis","entropy","class"]
df = pd.read_csv("data_banknote_authentication.txt", index_col=False, names=columns)
fig, ax = plt.subplots(1,4, figsize=(10,3))

# 特征的样本分布
vis1 = sns.distplot(df["var"],bins=10,ax= ax[0])
vis2 = sns.distplot(df["skewness"],bins=10,ax= ax[1])
vis3 = sns.distplot(df["curtosis"],bins=10,ax= ax[2])
vis4 = sns.distplot(df["entropy"],bins=10,ax= ax[3])
#fig.savefig('subplot.png')

# 不同特征维度的数据分布
sns.pairplot(df, hue="class")

#  计算每个类别的均值向量
class_mean_vec = []
for i in df["class"].unique():
    # pd.mean()返回每一列的均值
    class_mean_vec.append(np.array(df[df["class"]==i].mean()[:4]))

# 计算Sw和Sb
Sw = np.zeros((4,4))
for i in range(2):
    per_class_sc_mat = np.zeros((4,4))                                      # 每个类别的协方差矩阵
    for j in df[df["class"]==i].index:
        row, mv = np.array(df.loc[j,:"entropy"]).reshape(4,1),class_mean_vec[i].reshape(4,1)
        per_class_sc_mat += np.dot((row-mv),((row-mv).T))
    Sw += per_class_sc_mat                                                  #每个类别协方差之和
print('within-class Scatter Matrix:\n', Sw)
feature_mean_vec = np.array(df.drop("class", axis=1).mean())
Sb = np.zeros((4,4))
for i in range(2):
    n = df[df["class"]==i].shape[0]
    mv = class_mean_vec[i].reshape(4,1)
    overall_mean = feature_mean_vec.reshape(4,1)
    Sb += np.dot(n*(mv-overall_mean),(mv-overall_mean).T)
print('between-class Scatter Matrix:\n', Sb)

# 求解Sw^{-1}Sb的广义特征值问题，以获得线性判别式。
e_vals, e_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

eig_pairs = [(np.abs(e_vals[i]), e_vecs[:,i]) for i in range(len(e_vals))]
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)



# 选择对应于top-k特征值的top-k特征向量
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))

#W = np.stack((e_vecs[0],e_vecs[1]))
print('Matrix W:\n', W.real)

X = df[["var","skewness","curtosis","entropy"]]
X_lda = X.dot(W)
df["PC1"] = X_lda[[0]]
df["PC2"] = X_lda[[1]]

vis = sns.lmplot(data = df[["PC1","PC2","class"]], x = "PC1", y = "PC2",fit_reg=False, hue = "class",
                 size = 6, aspect=1.5, scatter_kws = {'s':50}, )
sns.pairplot(df[["PC1","PC2","class"]], hue="class")

plt.show()
