# -*- coding: utf-8 -*-

from sklearn import svm
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
from anti_input import anti_process_train,anti_process_test

anti_data, anti_target,ratio_worest = anti_process_train()
test_data , test_ids= anti_process_test(ratio_worest)
clf = svm.SVC(kernel="linear",C=1)
scores = cross_validate(clf,anti_data, anti_target,cv=2,return_train_score=False)

predict_label =clf.predict(test_data)
# for i in np.arange(99):
#     X = anti_data[1 + i * 10000:(i + 1) * 10000]
#     y = anti_target[1 + i * 10000:(i + 1) * 10000]
#     print(i,len(X),len(y))
#     clf.fit(X,y)
# frame = pd.DataFrame({'id':test_ids,
#                      'score':predict_label})
#
# frame.to_csv('/home/lin/Data/anti/result.csv',index=False)