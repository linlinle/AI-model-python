# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
import preprocessing_Anti

def anti_process_train():
    # 读入
    csv_train = pd.read_csv('/home/lin/Data/anti/atec_anti_fraud_train.csv/atec_anti_fraud_train_0.csv')
    #   去除缺失值较多的特征
    dataframe, ratio_worest = preprocessing_Anti.delet_train_mostnan(csv_train)
    csv_train = preprocessing_Anti.remove_missing_values(csv_train)
    csv_train, csv_label = preprocessing_Anti.category_data_balance(csv_train)
    csv_train = preprocessing_Anti.fearure_scaling(csv_train)
    return csv_train, csv_label.reset_index(drop=True),ratio_worest


def anti_process_test(ratio_worest):
    # 读入
    csv_test = pd.read_csv('/home/lin/Data/anti/atec_anti_fraud_test_a.csv/atec_anti_fraud_test_a.csv')
    csv_test, ids = preprocessing_Anti.get_ids_puredata(csv_test)
    csv_test = preprocessing_Anti.delet_test_mostnan(csv_test,ratio_worest)
    csv_test = preprocessing_Anti.replace_testnan(csv_test)
    # 填写缺失值
    scaler = StandardScaler()
    processed_test = scaler.fit_transform(csv_test)
    return processed_test, ids
