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
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

########################################################
# MODEL SETUP

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

################### NEURAL NET ###################
print 'NEURAL NET: '
from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier()

# # # scores = cross_val_score(mlp, X_res, y_res, scoring='recall', cv=5)
# # # print scores
# # # print 'Recall mean = ', np.mean(scores)

# mlp.fit(X_res, y_res)
# y_pred = mlp.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)
# class_names = [0,1]

# # plt.figure()
# # plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
# # plt.show()


######## GRID SEARCH STUFF
mlp = MLPClassifier(solver='sgd')
param_grid = { "hidden_layer_sizes"      : [(100,),(250,), (300,)],
           "activation"         : ['identity', 'logistic', 'tanh', 'relu'],
           "learning_rate"      : ['constant', 'adaptive']}

from sklearn.metrics import recall_score, make_scorer, f1_score
scorer = make_scorer(f1_score, pos_label=1)

# Also tried scoring='f1', scoring='recall' etc...
grid_search = GridSearchCV(mlp, param_grid, n_jobs=1, cv=3, scoring=scorer, verbose=50)
grid_search.fit(X_res, y_res)
print grid_search.best_params_, grid_search.best_estimator_

y_pred = grid_search.predict(X_test)
########

from sklearn.metrics import classification_report
print classification_report(y_test, y_pred)
print 'Test recall score: ', recall_score(y_test, y_pred)

### SAVE MODEL ###
import pickle

tuple_objects = (grid_search.best_estimator_, X_test, y_test)

# Save tuple
pickle.dump(tuple_objects, open("neural_network_tuned.pkl", 'wb'))


print 'Model saved.'



