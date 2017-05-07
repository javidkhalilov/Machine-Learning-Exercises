# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 19:13:48 2017

@author: Javid
"""


def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    from sklearn.naive_bayes import  GaussianNB
    clf=GaussianNB()
    return clf.fit(features_train,labels_train)
    
    ### your code goes here!