# coding: utf-8
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_uniform
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

batch_size = 128
epochs = 100
learnRate = 0.01

# Learning rate annealing
def step_decay(epoch):
    if epoch/epochs<0.3:
        lrate = learnRate
    elif epoch/epochs<=0.5:
        lrate = learnRate/2
    elif epoch/epochs<=0.70:
        lrate = learnRate/10
    else:
        lrate = learnRate/100
    return lrate


"""Load Data"""
img_rows, img_cols = 24, 24
faces = np.loadtxt('data/dfFaces_24x24_norm')
notfaces = np.loadtxt('data/NotFaces_24x24_norm')
yfaces = np.ones(faces.shape[0])
yNotfaces = np.zeros(notfaces.shape[0])

y = np.append(yfaces, yNotfaces)
x = np.concatenate((faces, notfaces), axis=0)

np.random.seed(1992)
aux_list = list(zip(x, y))
random.shuffle(aux_list)
x, y = zip(*aux_list)

x = np.array(x)
y = np.array(y)

x = x.reshape(x.shape[0], img_rows, img_cols, 1)

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1992)

print(x_train.shape)
print(x_test.shape)

"""Data Generator"""

datagen = ImageDataGenerator(width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=False,
                            vertical_flip=True)
datagen.fit(x_train)

"""Create Model for training"""
shape = x_train.shape[1:]

model = Sequential()
model.add(BN(input_shape=shape))
model.add(Dropout(0.1))
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer=glorot_uniform(seed=0)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BN())
model.add(GN(0.3))

model.add(Flatten())
model.add(Dense(256, kernel_initializer=glorot_uniform(seed=0)))
model.add(Activation('relu'))
model.add(BN())
model.add(Dropout(0.1))

model.add(Dense(128, kernel_initializer=glorot_uniform(seed=0)))
model.add(Activation('relu'))
model.add(BN())
model.add(Dropout(0.1))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

adam = Adam(lr=learnRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
lrate = LearningRateScheduler(step_decay)

model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

history = model.fit_generator(datagen.flow(x_train, y_train),
                    steps_per_epoch = len(x_train)/batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[lrate])

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', (1-score[1])*100,'%')

y_pred = model.predict(x_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
print(y_pred)
print("")
print("_________________Test Confusion Matrix_________________")
print(confusion_matrix(y_test, y_pred))
print("______________________Test Report______________________")
print(classification_report(y_test, y_pred))

model.save('model.h5')
model.save_weights('weights.h5')
