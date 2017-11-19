import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

data = pd.read_csv("creditcard.csv")

# Normalise and reshape the Amount column, so it's values lie between -1 and 1
from sklearn.preprocessing import StandardScaler
data['norm_Amount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))

# Drop the old Amount column and also the Time column as we don't want to include this at this stage
data = data.drop(['Time', 'Amount'], axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

################### LOGISTIC REGRESSION ###################
print 'LOGISTIC REGRESSION: '
# Call the logistic regression model with a certain C parameter
lr = LogisticRegression(C = 10)

# Assign variables x and y corresponding to row data and it's class value
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']


# Whole dataset, training-test data splitting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

from collections import Counter
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=1)
X_res, y_res = sm.fit_sample(X_train, y_train)
print('Original dataset shape {}'.format(Counter(data['Class'])))
print('Training dataset shape {}'.format(Counter(y_train['Class'])))
print('Resampled training dataset shape {}'.format(Counter(y_res)))

# # CROSS VALIDATION
# scores = cross_val_score(lr, X_res, y_res, scoring='recall', cv=5)
# print scores
# print 'Recall mean = ', np.mean(scores)

# lr.fit(X_res, y_res)
# y_pred = lr.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]


# plt.figure()
# plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# plt.show()

# from sklearn.metrics import classification_report
# print classification_report(y_test, y_pred)

# ################## K NEAREST NEIGHBORS ###################
# print 'K NEAREST NEIGHBORS: '
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier()

# # CROSS VALIDATION
# scores = cross_val_score(neigh, X_res, y_res, scoring='recall', cv=5)
# print scores
# print 'Recall mean = ', np.mean(scores)

# neigh.fit(X_res, y_res)
# y_pred = neigh.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]


# plt.figure()
# plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# plt.show()

# from sklearn.metrics import classification_report
# print classification_report(y_test, y_pred)

# ################### DECISION TREE ###################
# print 'DECISION TREE: '
# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()

# # scores = cross_val_score(dt, X_res, y_res, scoring='recall', cv=5)
# # print scores
# # print 'Recall mean = ', np.mean(scores)

# dt.fit(X_res, y_res)
# y_pred = dt.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]


# # plt.figure()
# # plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# # plt.show()

# from sklearn.metrics import classification_report
# print classification_report(y_test, y_pred)
# print recall_score(y_test, y_pred)

################### NEURAL NET ###################
# print 'NEURAL NET: '
# from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(200,100))

# # scores = cross_val_score(mlp, X_res, y_res, scoring='recall', cv=5)
# # print scores
# # print 'Recall mean = ', np.mean(scores)

# mlp.fit(X_res, y_res)
# y_pred = mlp.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]


# # plt.figure()
# # plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# # plt.show()

# from sklearn.metrics import classification_report
# print classification_report(y_test, y_pred)
# print recall_score(y_test, y_pred)


# ################### GAUSSIAN PROCESS ###################
# print 'GAUSSIAN PROCESS: '
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# gp = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
#                                    optimizer=None)

# # scores = cross_val_score(mlp, X_res, y_res, scoring='recall', cv=5)
# # print scores
# # print 'Recall mean = ', np.mean(scores)

# gp.fit(X_res, y_res)
# y_pred = gp.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]


# # plt.figure()
# # plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# # plt.show()

# from sklearn.metrics import classification_report
# print classification_report(y_test, y_pred)
# print recall_score(y_test, y_pred)

# ################### LINEAR SVM ###################
# print 'LINEAR SVM: '
# from sklearn.svm import SVC

# lin_svm = SVC()

# # scores = cross_val_score(mlp, X_res, y_res, scoring='recall', cv=5)
# # print scores
# # print 'Recall mean = ', np.mean(scores)

# lin_svm.fit(X_res, y_res)
# y_pred = lin_svm.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]


# # plt.figure()
# # plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# # plt.show()

# from sklearn.metrics import classification_report
# print classification_report(y_test, y_pred)
# print recall_score(y_test, y_pred)

################### RANDOM FOREST ###################
# print 'RANDOM FOREST: '
# from sklearn.ensemble import RandomForestClassifier

# rf = RandomForestClassifier()

# # scores = cross_val_score(mlp, X_res, y_res, scoring='recall', cv=5)
# # print scores
# # print 'Recall mean = ', np.mean(scores)

# rf.fit(X_res, y_res)
# y_pred = rf.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]


# model = RandomForestRegressor(random_state=30)
# param_grid = { "n_estimators"      : [250, 500, 750],
#            "criterion"         : ["gini", "entropy"],
#            "max_features"      : [3, 5],
#            "max_depth"         : [10, 20]}
# grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, cv=3)
# grid_search.fit(X, y)
# print grid_search.best_params_

# # plt.figure()
# # plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# # plt.show()

# from sklearn.metrics import classification_report
# print classification_report(y_test, y_pred)
# print recall_score(y_test, y_pred)

################### NAIVE BAYES ###################
# print 'Gaussian Naive Bayes: '

# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()

# scores = cross_val_score(gnb, X_res, y_res, scoring='recall', cv=5)
# print scores
# print 'Recall mean = ', np.mean(scores)

# gnb.fit(X_res, y_res)
# y_pred = gnb.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]

# plt.figure()
# plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# plt.show()

# from sklearn.metrics import classification_report
# print classification_report(y_test, y_pred)
# print recall_score(y_test, y_pred)

################### GAUSSIAN PROCESS ###################
print 'Gaussian Process: '

from sklearn.gaussian_process import GaussianProcessClassifier
gpc = GaussianProcessClassifier()

scores = cross_val_score(gpc, X_res, y_res, scoring='recall', cv=5)
print scores
print 'Recall mean = ', np.mean(scores)

gpc.fit(X_res, y_res)
y_pred = gpc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
class_names = [0,1]

plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
plt.show()

from sklearn.metrics import classification_report
print classification_report(y_test, y_pred)
print recall_score(y_test, y_pred)


