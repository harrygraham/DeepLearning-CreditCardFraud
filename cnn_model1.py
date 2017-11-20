from numpy.random import seed
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
# from keras_diagram import ascii
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

data = pd.read_csv("creditcard.csv")

# Normalise and reshape the Amount column, so it's values lie between -1 and 1
from sklearn.preprocessing import StandardScaler
data['norm_Amount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))

# Drop the old Amount column and also the Time column as we don't want to include this at this stage
data = data.drop(['Time', 'Amount'], axis=1)

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

X_res, y_res, X_test, y_test = generate_train_test_sample(X, y)

print X_res.shape, type(X_res)
print y_res.shape

X_train = X_res.reshape(X_res.shape[0], 29, 1)
Y_train = y_res.reshape(y_res.shape[0], 1)
X_test = X_test.values.reshape(X_test.values.shape[0], 29, 1)
Y_test = y_test.values.reshape(y_test.values.shape[0], 1)

seed(2017)
conv = Sequential()
conv.add(Conv1D(32, 2, input_shape=(29, 1), activation='relu'))
conv.add(MaxPooling1D())
conv.add(Flatten())
# conv.add(Conv2D(32, (1, 4), input_shape = (None, 29), activation = 'relu'))
# conv.add(Flatten())
conv.add(Dense(300, activation = 'relu'))
conv.add(Dense(1, activation = 'sigmoid'))

sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
conv.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
conv.fit(X_train, Y_train, batch_size = 500, epochs = 15, verbose = 50)
score = conv.evaluate(X_test, Y_test, batch_size=500)

y_pred = conv.predict(X_test)

prfs = precision_recall_fscore_support(Y_test, y_pred.round(), pos_label=1, average='binary')
print prfs 

########################################################################

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.layers import Embedding
# from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

# model = Sequential()
# model.add(Conv1D(10, 2, activation='relu', input_shape=(29, 1)))
# model.add(MaxPooling1D())
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(1,activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# model.fit(X_res, y_res, batch_size=16, epochs=10, verbose =50)
# score = model.evaluate(X_test, y_test, batch_size=16)


# # conv = Sequential()
# # conv.add(Convolution1D(64, 10, input_shape=(1,101)))
# # conv.add(Activation('relu'))
# # conv.add(MaxPooling1D(2))
# # conv.add(Flatten())
# # conv.add(Dense(10))
# # conv.add(Activation('tanh'))
# # conv.add(Dense(4))
# # conv.add(Activation('softmax'))

# np.random.seed(1337) # for reproducibility
 
# import os
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
 
# batch_size = 128
# nb_classes = 2
# nb_epoch = 12
 
# # input image dimensions
# img_rows, img_cols = 1, 29
# # number of convolutional filters to use
# nb_filters = 32
# # size of pooling area for max pooling
# nb_pool = 2
# # convolution kernel size
# nb_conv = 3

#Add the depth in the input. Only grayscale so depth is only one
#see http://cs231n.github.io/convolutional-networks/#overview

 
# #Display the shapes to check if everything's ok
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
 
# model = Sequential()
# #For an explanation on conv layers see http://cs231n.github.io/convolutional-networks/#conv
# #By default the stride/subsample is 1
# #border_mode "valid" means no zero-padding.
# #If you want zero-padding add a ZeroPadding layer or, if stride is 1 use border_mode="same"
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
# border_mode='valid',
# input_shape=(1, img_rows, img_cols)))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
# #For an explanation on pooling layers see http://cs231n.github.io/convolutional-networks/#pool
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))
# #Flatten the 3D output to 1D tensor for a fully connected layer to accept the input
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes)) #Last layer with one output per class
# model.add(Activation('softmax')) #We want a score simlar to a probability for each class
 
# #The function to optimize is the cross entropy between the true label and the output (softmax) of the model
# #We will use adadelta to do the gradient descent see http://cs231n.github.io/neural-networks-3/#ada
# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
 
# #Make the model learn
# model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
# verbose=1, validation_data=(X_test, Y_test))
 
# #Evaluate how the model does on the test set
# score = model.evaluate(X_test, Y_test, verbose=0)
 
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# X_res = np.expand_dims(X_res, axis=0)
# # print X_res.shape