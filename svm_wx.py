# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:53:57 2017

@author: Javid
"""

import sys
from class_vis import prettyPicture,output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(C=1.0,kernel="rbf")
clf.fit(features_train,labels_train)

pred=clf.predict(features_test)
#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data



#### store your predictions in a list named pred

prettyPicture(clf, features_test, labels_test)
plt.show()




from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print(acc)
def submitAccuracy():
    return acc