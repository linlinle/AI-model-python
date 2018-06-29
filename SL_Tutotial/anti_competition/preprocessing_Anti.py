# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def remove_missing_values(dataframe):
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    dataframe.ix[:, 3:] = imr.fit_transform(dataframe.ix[:,3:])
    return dataframe
def replace_testnan(dataframe):
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    dataframe = imr.fit_transform(dataframe)
    return dataframe

def category_data_balance(dataframe):
    class_uncertain = dataframe[dataframe['label'] == -1]
    class_0 = dataframe[dataframe['label'] == 0]
    class_1 = dataframe[dataframe['label'] == 1]
    class_0 = class_0.sample(frac=0.015)

    pure_data_0 ,target_0 = get_pure_data(class_0)
    pure_data_1 ,target_1 = get_pure_data(class_1)
    pure_data_uncertain ,target_uncertain = get_pure_data(class_uncertain)

    target_uncertain = cluster_uncertain(pure_data_uncertain,pd.concat([pure_data_0,pure_data_1]))

    return pd.concat([pure_data_0,pure_data_1,pure_data_uncertain]),pd.concat([target_0,target_1,target_uncertain])

def get_pure_data(dataframe):
    label = dataframe['label']
    dataframe.drop(['id','label','date',], axis=1,inplace=True)
    return dataframe,label

def fearure_scaling(dataframe):
    scaler = StandardScaler()
    processed_train = scaler.fit_transform(dataframe)
    return processed_train

def cluster_uncertain(df_uncertain, data):
    clf = KMeans(n_clusters=2,random_state=0).fit(data)
    uncertain_target = clf.predict(df_uncertain)
    return pd.Series(uncertain_target)

def get_ids_puredata(dataframe):
    ids = dataframe['id']
    dataframe.drop(['id', 'date'], axis=1, inplace=True)
    return dataframe,ids

def delet_train_mostnan(dataframe):

        ratio_nan = dataframe.isnull().sum(axis=0) / dataframe.__len__()
        ratio_order = ratio_nan.sort_values()
        ratio_worest = ratio_order[ratio_order > 0.8]
        dataframe.drop(ratio_worest.index, axis=1, inplace=True)
        return dataframe, ratio_worest
def delet_test_mostnan(dataframe, ratio):
    #   去除缺失值较多的特征
    dataframe.drop(ratio.index, axis=1, inplace=True)
    return dataframe
