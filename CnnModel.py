''' Import theano and numpy '''
import tensorflow
import numpy as np

''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

#Load data
train_data = np.load('Classical_Data_Pack.txt')
shape = train_data[0].shape
tag = np.genfromtxt('classical_tag.txt', delimiter=',')
tag = tag.astype('int')
tag = np_utils.to_categorical(tag,8)

''' For categorical_crossentropy '''
model_ce = Sequential()
model_ce.add(Convolution2D(64,3,3, input_shape= shape))
model_ce.add(MaxPooling2D(pool_size = (2,2)))
model_ce.add(Convolution2D(64,3,3))
model_ce.add(MaxPooling2D(pool_size = (2,2)))
model_ce.add(Convolution2D(64,3,3))
model_ce.add(MaxPooling2D(pool_size = (2,2)))
model_ce.add(Convolution2D(64,3,3))
model_ce.add(MaxPooling2D(pool_size = (2,2)))
model_ce.add(Convolution2D(64,3,3))
model_ce.add(MaxPooling2D(pool_size = (2,2)))
model_ce.add(Flatten())
model_ce.add(Dense(256))
model_ce.add(Dense(8))
model_ce.add(Activation('softmax'))

''' Set up the optimizer '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.0017,momentum=0.0,decay=0.0,nesterov=False)

''' Compile model with specified loss and optimizer '''
model_ce.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])

''' set the size of mini-batch and number of epochs'''
batch_size = 16
nb_epoch = 32

'''Fit models and use validation_split=0.1 '''
history_ce = model_ce.fit(train_data, tag,
							verbose=0,
							shuffle=True,
							nb_epoch = nb_epoch,
                    		validation_split=0.2)

'''Access the loss and accuracy in every epoch'''
loss_ce	= history_ce.history.get('loss')
acc_ce 	= history_ce.history.get('acc')
val_loss = history_ce.history.get('val_loss')
val_acc = history_ce.history.get('val_acc')
''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure('Cnn')
plt.subplot(121)
plt.plot(range(len(loss_ce)), loss_ce,label='Training')
plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_ce)), acc_ce,label='Training')
plt.plot(range(len(val_acc)), val_acc,label='Validation')
plt.title('Accuracy')
plt.savefig('CNN.png',dpi=300,format='png')
plt.show()
