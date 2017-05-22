''' Import theano and numpy '''

import numpy as np

''' Import keras to build a DL model '''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

train_data_dir = 'SmallPic/'

''' For categorical_crossentropy '''
model_ce = Sequential()
model_ce.add(Conv2D(64,(11,11), input_shape=(256,256,3) ))
model_ce.add(Activation('relu'))
model_ce.add(MaxPooling2D(pool_size = (2,2)))
model_ce.add(Conv2D(64,(5,5)))
model_ce.add(Activation('relu'))
model_ce.add(MaxPooling2D(pool_size = (2,2)))
model_ce.add(Conv2D(64,(3,3)))
model_ce.add(Activation('relu'))
model_ce.add(MaxPooling2D(pool_size = (2,2)))
model_ce.add(Flatten())
model_ce.add(Dense(256))
model_ce.add(Dense(128))
model_ce.add(Dense(4))
model_ce.add(Activation('softmax'))

''' Set up the optimizer '''
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
sgd = SGD(lr=0.00017,momentum=0.0,decay=0.0,nesterov=False)

''' Compile model with specified loss and optimizer '''
model_ce.compile(loss='categorical_crossentropy',
				optimizer=sgd,
				metrics=['accuracy'])

''' set the size of mini-batch and number of epochs'''
batch_size = 16
nb_epoch = 16

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    batch_size=16)

history_ce = model_ce.fit_generator(train_generator,steps_per_epoch=10,epochs=nb_epoch,verbose = 1, validation_data=None, validation_steps=None)

'''
history_ce = model_ce.fit(train_data, tag,
							verbose=0,
							shuffle=True,
							nb_epoch = nb_epoch,
                    		validation_split=0.1)
'''
'''Access the loss and accuracy in every epoch'''
loss_ce	= history_ce.history.get('loss')
acc_ce 	= history_ce.history.get('acc')
print (loss_ce)
print (acc_ce)
'''
val_loss = history_ce.history.get('val_loss')
val_acc = history_ce.history.get('val_acc')
'''
''' Visualize the loss and accuracy of both models'''
import matplotlib.pyplot as plt
plt.figure('Cnn')
plt.subplot(121)
plt.plot(range(len(loss_ce)), loss_ce,label='Training')
#plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc_ce)), acc_ce,label='Training')
#plt.plot(range(len(val_acc)), val_acc,label='Validation')
plt.title('Accuracy')
plt.savefig('CNN.png',dpi=300,format='png')
plt.show()
