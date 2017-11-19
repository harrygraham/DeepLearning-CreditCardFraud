import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

data = pd.read_csv("creditcard.csv")

# Normalise and reshape the Amount column, so it's values lie between -1 and 1
from sklearn.preprocessing import StandardScaler
data['norm_Amount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))

# Drop the old Amount column and also the Time column as we don't want to include this at this stage
data = data.drop(['Time', 'Amount'], axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score, precision_score, f1_score, classification_report, make_scorer, precision_recall_fscore_support


# Assign variables x and y corresponding to row data and it's class value
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']


# Whole dataset, training-test data splitting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

from collections import Counter
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X_res, y_res = sm.fit_sample(X, y)
print('Original dataset shape {}'.format(Counter(data['Class'])))
print('Training dataset shape {}'.format(Counter(y_train['Class'])))
print('Resampled training dataset shape {}'.format(Counter(y_res)))

########################################################################

# A sample toy binary classification dataset
from sklearn import datasets
from sklearn.svm import LinearSVC
X, y = datasets.make_classification(n_classes=2, random_state=0)
svm = LinearSVC(random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

# Functions for individual confusion matrix elements
def tpr(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tnr(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fpr(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fnr(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

def f1r(y_true, y_pred): 
    tp = tpr(y_true, y_pred)
    tn = tnr(y_true, y_pred)
    fp = fpr(y_true, y_pred)
    fn = fnr(y_true, y_pred)

    precision = tp / float((tp + fp))
    recall = tp / float((tp + fn))
    f1 = ((2 * (precision * recall)) / ((precision + recall)))
    return f1

def f1r2(y_true, y_pred): 
    

    precision = tp / float((tp + fp))
    recall = tp / float((tp + fn))
    f1 = ((2 * (precision * recall)) / ((precision + recall)))
    return f1

# scoring = {'tp' : make_scorer(tpr), 'tn' : make_scorer(tnr),
#          'fp' : make_scorer(fpr), 'fn' : make_scorer(fnr), 'f1': make_scorer(f1_score)}

# scoring = {'f1': make_scorer(f1r)}
scoring = {'f1': make_scorer(f1_score)}

cv_results = cross_validate(svm, X, y, scoring='f1', verbose=50, cv=20)
# # Getting the test set true positive scores
# print(cv_results['test_tp'])          

# # Getting the test set false negative scores
# print(cv_results['test_fn'])          

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print cv_results
print np.mean(cv_results['test_score'])
print cm
print classification_report(y_test, y_pred)
print precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')[2]
