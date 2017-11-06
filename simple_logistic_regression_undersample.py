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

# Examine data
# print data.head()

# Print a plot of class balance
# classes = pd.value_counts(data['Class'], sort=True)
# classes.plot(kind = 'bar')
# plt.show()

# Normalise and reshape the Amount column, so it's values lie between -1 and 1
from sklearn.preprocessing import StandardScaler
data['norm_Amount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))

# Drop the old Amount column and also the Time column as we don't want to include this at this stage
data = data.drop(['Time', 'Amount'], axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

fraud_indices = np.array(data[data.Class == 1].index)
print 'Number of frauds', len(fraud_indices)

# data.sample(frac=1)
non_fraud = data[data.Class==0]
fraud = data[data.Class==1]

print 'number of non fraud: ', len(non_fraud)
non_fraud = non_fraud.loc[np.random.choice(non_fraud.index, len(fraud_indices), replace=False)]

data = pd.concat([non_fraud, fraud])
print 'non_fraud after: ', len(non_fraud)

# Call the logistic regression model with a certain C parameter
lr = LogisticRegression(C = 10)

# Assign variables x and y corresponding to row data and it's class value
X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']





# Whole dataset, training-test data splitting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

# CROSS VALIDATION
scores = cross_val_score(lr, X, y, scoring='recall', cv=5)
print scores
print 'Recall mean = ', np.mean(scores)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
class_names = [0,1]


plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
plt.show()

from sklearn.metrics import classification_report
print classification_report(y_test, y_pred)




