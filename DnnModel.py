''' Import theano and numpy '''
import tensorflow
import numpy as np

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


X_train = np.genfromtxt('Data/feature.txt',delimiter = ',')
Y_train = np.genfromtxt('Data/tag.txt')
Y_train = Y_train.astype('int')
Y_train = np_utils.to_categorical(Y_train,9)

print 'Building a model whose loss function is categorical_crossentropy'
''' For categorical_crossentropy '''
model_ce = Sequential()
model_ce.add(Dense(128, input_dim=3))
model_ce.add(Activation('softplus'))
model_ce.add(Dense(256))
model_ce.add(Activation('softplus'))
model_ce.add(Dense(512))
model_ce.add(Activation('softplus'))
model_ce.add(Dense(256))
model_ce.add(Activation('softplus'))
model_ce.add(Dense(9))
model_ce.add(Activation('softmax'))

''' Set up the optimizer '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.001,momentum=0.0,decay=0.0,nesterov=False)

''' Compile model with specified loss and optimizer '''
model_ce.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])

''' set the size of mini-batch and number of epochs'''
batch_size = 16
nb_epoch = 30

'''Fit models and use validation_split=0.1 '''
history_ce = model_ce.fit(X_train, Y_train,
							batch_size=batch_size,
							nb_epoch=nb_epoch,
							verbose=0,
							shuffle=True,
                    		validation_split=0.1)

'''Access the loss and accuracy in every epoch'''
loss_ce	= history_ce.history.get('loss')
acc_ce 	= history_ce.history.get('acc')

''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(121)
plt.plot(range(len(loss_ce)), loss_ce,label='CE')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_ce)), acc_ce,label='CE')
plt.title('Accuracy')
plt.show()
plt.savefig('00_firstModel.png',dpi=300,format='png')

print 'Result saved into 00_lossFuncSelection.png'
