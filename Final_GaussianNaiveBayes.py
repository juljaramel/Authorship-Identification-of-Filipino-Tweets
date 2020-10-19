# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:26:37 2020

@author: Ash
"""


import pandas as pd #the import is used for data frames
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Dataset Path
DATASET_PATH = r'C:\Users\Ash\Documents\Python Scripts\FE.csv'

# declaring of data header
features_headers = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","AUTHORS"]
features = pd.read_csv(DATASET_PATH, names=features_headers)
   


#DATASET
features.shape

#DISTRIBUTION OF CLASSES
features['AUTHORS'].value_counts()

#TARGET COLUMN AND TARGET CLASS
X = features.drop(['AUTHORS'],axis=1) #GETTING ALL THE COLUMNS EXCEPT FOR AUTHORS
y = features['AUTHORS'] # GETTING ONLY THE AUTHORS COLUMN

#SPLITTING DATA 80% TRAIN, 20% TEST

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#GAUSSIAN NAIVE BAYES IMPLEMENTATION
gnb = GaussianNB()

#FITTING OF THE TRAINING DATA
gnb.fit(X_train,y_train)


#DECLARING VARIABLE FOR TEST DATA
y_pred = gnb.predict(X_test)

#DECLARING VARIABLE FOR PLOTTING THE MATRIX
cm = confusion_matrix(y_test, y_pred)

#DECLARING CONFUSION MATRIX VARIABLES
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

#FORMULAS AND COMPUTATIONS
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
precision = TP / float(TP + FP)
recall = TP / float(TP + FN)
specificity = TN / (TN + FP)
f1 = ((float(precision) + float(recall) )/ 2)


# PRINTING THE DATA SHAPES
print("Data Shape: ",features.shape)
print("Training Shape: ",X_train.shape, "\nTesting Shape",X_test.shape)

#plotting confusion matrix and printing
print('\nConfusion matrix: \n', cm)
print('\nTrue Positives(TP) = ', TP)
print('True Negatives(TN) = ', TN)
print('False Positives(FP) = ', FP)
print('False Negatives(FN) = ', FN)

#PRINTING OF THE RESULTS
print('\nClassification accuracy : {0:0.2f}'.format(classification_accuracy))
print('Precision : {0:0.2f}'.format(precision))
print('Recall : {0:0.2f}'.format(recall))
print('Specificity : {0:0.2f}'.format(specificity))
print('f1-score : {0:0.2f}'.format(f1))