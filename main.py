import pandas as pd
import numpy as np
import os
from McOne import McOne
from McTwo import McTwo
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

def evaluation(X, y):
    y = y.astype('int')
    kf = KFold(n_splits = 5)
    mAcc = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        acc1 = np.mean(SVC().fit(X_train, y_train).predict(X_test) == y_test)
        acc2 = np.mean(GaussianNB().fit(X_train, y_train).predict(X_test) == y_test)
        acc3 = np.mean(DecisionTreeClassifier().fit(X_train, y_train).predict(X_test) == y_test)
        acc4 = np.mean(KNeighborsClassifier().fit(X_train, y_train).predict(X_test) == y_test)
        mAcc.append(max(acc1, acc2, acc3, acc4))
    return np.array(mAcc).mean()

for dataName in os.listdir('./data/'):
    print(f'Dataset: {dataName}')
    
    data = pd.read_table(os.path.join('./data/', dataName), header = None, index_col=0, low_memory=False).transpose().to_numpy()
    feature = data[:, 1:]
    label = data[:, 0]
    for index, l in enumerate(list(set(label))):
        label[np.where(label == l)] = index
    
    FOne = McOne(feature, label, 0.2)
    FTwo = McTwo(FOne, label)
    print(f'FOne.shape: {FOne.shape}, FTwo.shape: {FTwo.shape}')
    
    mAcc1 = evaluation(FOne, label)
    mAcc2 = evaluation(FTwo, label)
    print(f'mAcc1: {mAcc1}, mAcc2: {mAcc2}')