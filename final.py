# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:42:50 2020

@author: Saeed Lotfi
"""
#importing dataset and packages
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


digits = load_digits()

#defining function
def nmi(digits):
    #normalizing data
    x = MinMaxScaler().fit_transform(digits.data)
    x = pd.DataFrame(x)
    y = pd.DataFrame(digits.target)
    #splitting data to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    a = x.values
    b = y.values
    b = b.flatten()
    S = []
    k = []
    
    #clustering each feature and calculating NMI 
    for i in range(a.shape[1]):
        km = MiniBatchKMeans(init='k-means++', n_clusters=np.unique(b).shape[0],batch_size=50)
        p = km.fit_predict(a[:,i].reshape(-1,1))
        s = normalized_mutual_info_score(b,p)
        S.append(s)
        k.append(i)
    s = pd.DataFrame(S).sort_values(by = 0,ascending=False )
    l = s.index.values.astype(int) 
    
    #initializing SVC
    svclassifier = SVC(kernel='linear')
    
    #model fitting with All features
    print("\nRESULT FOR ALL FEATURES \n")
    
    #fitting the model with the selected features
    svclassifier.fit(x_train , np.ravel(y_train))
    y_pred = svclassifier.predict(x_test)
    acc=accuracy_score(np.ravel(y_test), y_pred)
    print("\nClassification: \n")
    print(classification_report(y_test,y_pred, digits=4))
    print("-----------------------------------------------------\n")
    
    #model fitting with the selected features KNMIFI
    x_train1 = x_train.iloc[:,l[0]]
    x_test1 = x_test.iloc[: ,l[0]]
    l1 = []
    l1.append(l[0])
    prev=0    
    for i in range(1,len(l)):
        x_train1 = pd.concat([x_train1 , x_train.iloc[:,l[i]]] , axis=1)
        x_test1 = pd.concat([x_test1 , x_test.iloc[:,l[i]]] , axis=1)
        svclassifier.fit(x_train1 , np.ravel(y_train))
        y_pred = svclassifier.predict(x_test1)
        acc=accuracy_score(np.ravel(y_test), y_pred)
        if acc > prev:
            if(i != len(l)-1):
                l1.append(l[i])
                prev = acc
            else:
                print(l1)
        else:
            x_train1 = x_train1.drop([l[i]] , axis=1)
            x_test1 = x_test1.drop([l[i]] , axis=1)
            del l1[-1]
            if(i != len(l)-1):
                l1.append(l[i])
                
    print("RESULT FOR KNMIFI")
    svclassifier.fit(x_train1 , np.ravel(y_train))
    y_pred = svclassifier.predict(x_test1)
    acc=accuracy_score(np.ravel(y_test), y_pred)
    print("\nClassification: \n")
    print(classification_report(y_test,y_pred, digits=4))
    print("\nNUMBER OF FEATURES SELECTED: " + str(len(l1)))
    print("\n-----------------------------------------------------\n")
    
    #model fitting with the selected features KNMILFE
    print("RESULT FOR KNMILFE")
    s = pd.DataFrame(S).sort_values(by = 0,ascending=True )
    l = s.index.values.astype(int)
    l2 = []
    l3 = []
    l2.append(l[0])
    acc=0
    prev=0
    x_train2 = x_train.iloc[:,l[0]]
    x_test2 = x_test.iloc[:,l[0]]
    # adding all the feautres in the initial list
    for i in range(1,len(l)):
        l2.append(l[i])
        x_train2 = pd.concat([x_train2 , x_train.iloc[:,l[i]]] , axis=1)
        x_test2 = pd.concat([x_test2 , x_test.iloc[:,l[i]]] , axis=1)
        
    for i in range(len(l2)-1):
        svclassifier.fit(x_train2 , np.ravel(y_train))
        y_pred = svclassifier.predict(x_test2)
        acc=accuracy_score(np.ravel(y_test), y_pred)
        l3.append(acc)
        #selecting the best accuracy
        if acc > prev:
            prev=acc
            j = i
        del l2[0]
        x_train2 = x_train[l2]
        x_test2 = x_test[l2]
    svclassifier.fit(x_train2 , np.ravel(y_train))
    y_pred = svclassifier.predict(x_test2)
    acc=accuracy_score(np.ravel(y_test), y_pred)
    l3.append(acc)
    l2 = l[j:]
    x_train2 = x_train[l2]
    x_test2 = x_test[l2]
    #fitting the model with the least ranked features eliminated
    svclassifier.fit(x_train2 , np.ravel(y_train))
    y_pred = svclassifier.predict(x_test2)
    acc=accuracy_score(np.ravel(y_test), y_pred)
    print("\nClassification: \n")
    print(classification_report(y_test,y_pred, digits=4))
    print("\nNUMBER OF FEATURES SELECTED: " + str(len(l2)))
    
    #plotting Feature ranking
    plt.figure(1)
    plt.bar(k, S)
    plt.xlabel('Feature Fame', fontweight ='bold') 
    plt.ylabel('Ranking Values', fontweight ='bold')
    plt.title("Feature ranking" , fontweight ='bold')
    plt.show()
    
    #plotting Change in accuracy in the KNFE method
    plt.figure(2)
    plt.bar(k, l3)
    plt.xlabel('Number Of Feature Eliminated', fontweight ='bold') 
    plt.ylabel('Accuracy Percentage', fontweight ='bold')
    plt.title("Change in accuracy in the KNFE method" , fontweight ='bold')
    plt.show()
    
nmi(digits)