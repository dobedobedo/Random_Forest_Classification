#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:08:22 2018

@author: uqytu1
"""

import pandas as pd
import numpy as np
import itertools
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics

inputfile = '/home/uqytu1/Documents/spray_assessment_colour.xlsx'

def ParseInputNumbers(selects):
    try:
        selected_index = set()
        for select in selects:
            if '-' in select:
                start, end = select.split('-')
                for i in range(int(start), int(end)+1):
                    selected_index.add(i-1)
                        
            else:
                selected_index.add(int(select)-1)
        return selected_index
    except ValueError:
        print('Can\'t recognise input. Please try again!')
        return False

def plot_confusion_matrix(cm, classes,
                          normalise=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalise=True`.
    """
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 26).set_y(1.05)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True health score', size = 20)
    plt.xlabel('Predicted health score', size = 20)
    
def plot_oob(X, Y, 
             title='Out-of-bag accuracy during iteration'):
    plt.plot(X, Y, 'r-', label='Out-of-bag accuracy')
    plt.grid(which='major', alpha=0.6, ls='-')
    plt.title(title, size = 26).set_y(1.05)
    plt.ylabel('Out-of-bag accuracy (%)', size = 20)
    plt.xlabel('Iteration time', size = 20)

if __name__ == '__main__':
    # Load the Excel file, with the first column as index and first row as header
    data = pd.read_excel(inputfile, index_col=0)
    
    # Print out the headers and ask users to select features
    Items = list(data.keys())
    for index, item in enumerate(Items):
        print('{:>2}: {:<}'.format(index+1, item))
    print('Input the feature(s) for random forest classification:')
    print('e.g.: "1-3" to select 1 to 3; 1 3 to select 1 and 3')
    
    while True:
        selects = input('>>> ').split()
        selected_index = ParseInputNumbers(selects)
        if selected_index:
            break
        else:
            continue
    
    selected_features = list()
    for index in selected_index:
        selected_features.append(Items[index])
    
    # Get the features and labels for random forest classifier and regressor
    Features = data[selected_features]
    
    # Ask user for training label
    print('Input the feature as training label')
    while True:
        try:
            Labels = data[Items[int(input('>>> ').split()[0]) - 1]]
            break
        except IndexError:
            continue
    
    # Create an adaptive random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, warm_start=True, n_jobs=-1)
    oob_score = list()
    Full_training_set = np.array([], dtype=np.int64)
    # Iterative random forest for 300 loops
    counter = 0
    for i in range(10000):
        # Set 80% training samples
        X_train, X_test, Y_train, Y_test=train_test_split(Features, Labels, test_size=0.2, random_state=i)
        if len(set(Y_train)) < 4 or len(set(Y_test)) < 4:
            continue
        
        # Append new subsamples to full training samples
        Full_training_set = np.append(Full_training_set, Y_train)
        
        # Calculate the class weights
        classes = np.unique(Full_training_set)
        weights = compute_class_weight('balanced', classes, Full_training_set)
        class_weight = dict()
        for _idx, class_ in enumerate(classes):
            class_weight[class_] = weights[_idx]
        
        clf.class_weight = class_weight
        
        # Train a random forest classifier
        clf.fit(X_train, Y_train)
        oob_score.append(clf.oob_score_)
        counter += 1
        if counter == 300:
            break
        clf.n_estimators += 100
        
    # Create classification prediction
    Y_clf_predict = clf.predict(Features)
    
    # Calculate model accuracy, out-of-bag score, and feature importance
    Model_accuracy = metrics.accuracy_score(Labels, Y_clf_predict)
    #oob_score = clf.oob_score_
    feature_imp = pd.Series(clf.feature_importances_,index=selected_features).sort_values(ascending=False)
    
    # Calculate confusion matrix
    cm = metrics.confusion_matrix(Labels, Y_clf_predict)
    
    # Print the feature importance
    print(feature_imp)
    
    # Plot
    plt.figure()
    plt.subplot(121)
    plot_confusion_matrix(cm, classes=[1, 2, 3, 4], normalise=False,
                      title='Model accuracy: {:.2f}%'.format(Model_accuracy*100))
    plt.subplot(122)
    plot_oob(np.linspace(1, len(oob_score), num=len(oob_score)), np.array(oob_score)*100)
    plt.show()
    