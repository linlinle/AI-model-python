# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('E:/baidu/datasets/train.txt',delim_whitespace=True,names=['filename','target'])
filename_train, filename_test, label_train, label_test = train_test_split(
    df['filename'],df['target'],test_size=0.2,random_state=0)
train_csv = pd.concat([filename_train,label_train-1],axis=1)
test_csv = pd.concat([filename_test,label_test-1],axis=1)
train_csv.to_csv('E:/baidu/datasets/preprocessing/train.txt',sep=' ',columns=None,index=False)
test_csv.to_csv('E:/baidu/datasets/preprocessing/val.txt',sep=' ',columns=None,index=False)
d_f = pd.read_csv('E:/baidu/datasets/test.txt')
d_f['0'] = 0
d_f.to_csv('E:/baidu/datasets/preprocessing/test.txt',sep=' ',index=False)

