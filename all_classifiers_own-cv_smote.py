import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime

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


def generate_train_test_sample(x_data, y_data): 
    ''' 1) Generate new, random train-test split
        2) Random smote oversample the train data, keeping test data unseen
        3) Use this new train-test split to fit and test model
    '''

    X_train, X_test, y_train, y_test = train_test_split(x_data,y_data,test_size = 0.3)

    from collections import Counter
    from imblearn.over_sampling import SMOTE
    sm = SMOTE()
    X_res, y_res = sm.fit_sample(X_train, y_train)
    print('Resampling the data with SMOTE. . .')
    print('Resampled training dataset shape {}'.format(Counter(y_res)))

    return X_res, y_res, X_test, y_test

########################################################################
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(random_state=0),
    MLPClassifier(),
    GaussianNB()]

# Setting up dataframe table properties
log_cols=["Classifier", "F1 Score", "Precision", "Recall", "Training Time"]
log = pd.DataFrame(columns=log_cols)
print("="*30)

# Loop over the classifiers, fit the data over 3 iterations, gather results, input to dataframe table
for clf in classifiers:
    precision = []
    recall = []
    f1score = []
    elapsed_times = []

    name = clf.__class__.__name__
    
    print(name)

    for x in range(0, 3):
        
        print('\n')
        print('ITERATION {}:'.format(x))

        X_res, y_res, X_test, y_test = generate_train_test_sample(X, y)

        print('Fitting the model...')
        start = datetime.datetime.now()
        clf.fit(X_res, y_res)
        end = datetime.datetime.now()
        elapsed = end - start
        elapsed_times.append(elapsed)

        y_pred = clf.predict(X_test)

        print('**** Results ****')
        report = classification_report(y_test, y_pred)
        print(report)

        prfs = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
        precision.append(prfs[0])
        recall.append(prfs[1])
        f1score.append(prfs[2])

    average_timedelta = sum(elapsed_times, datetime.timedelta(0)) / len(elapsed_times)
    entry = [name, np.mean(f1score), np.mean(precision), np.mean(recall), average_timedelta]
    print('Mean scores: ', entry )
    print("="*30)

    log_entry = pd.DataFrame([entry], columns=log_cols)
    log = log.append(log_entry)

# Replace table index by the Classifier column
log.set_index('Classifier', inplace=True)
print log 
print("="*30)

# Plots to visualise results
ax = log[['F1 Score']].plot(kind='bar', title ="F1 Score", figsize=(15, 10), legend=True, fontsize=24)
ax.set_xlabel("Classifier", fontsize=14)
ax.set_ylabel("Score", fontsize=14)
plt.show()

ax = log[['Recall']].plot(kind='bar', title ="Recall", figsize=(15, 10), legend=True, fontsize=24)
ax.set_xlabel("Classifier", fontsize=14)
ax.set_ylabel("Score", fontsize=14)
plt.show()

ax = log[['Precision']].plot(kind='bar', title ="Precision", figsize=(15, 10), legend=True, fontsize=24)
ax.set_xlabel("Classifier", fontsize=14)
ax.set_ylabel("Score", fontsize=14)
plt.show()

    



