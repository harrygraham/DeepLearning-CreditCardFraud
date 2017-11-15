import pickle 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

# Restore tuple
pickled_model, pickled_Xtest, pickled_Ytest = pickle.load(open("tuple_model.pkl", 'rb'))  

# Calculate the accuracy score and predict target values
score = pickled_model.score(pickled_Xtest, pickled_Ytest)  
print("Test score: {0:.2f} %".format(100 * score))  
Ypredict = pickled_model.predict(pickled_Xtest) 

from sklearn.metrics import classification_report
print classification_report(pickled_Ytest, Ypredict)
print pickled_model

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

cm = confusion_matrix(pickled_Ytest, Ypredict)
class_names = [0,1]


plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix')
plt.show()