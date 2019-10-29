from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Convolution2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.losses import categorical_crossentropy
from keras import regularizers
from keras.regularizers import l1
import pandas as pd
import numpy as np
import csv
import cv2

width, height = 48, 48
image_size = (width, height)
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
MODELPATH = './models/model.h5'

def process_csv_file(data):
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), image_size)
        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

dataset_df = pd.read_csv("./dataset/fer2013.csv")

mask_training = (dataset_df['Usage']=="Training")
mask_test = (dataset_df['Usage']=="PublicTest")
mask_valid = (dataset_df['Usage']=="PrivateTest")

train_raw = dataset_df.loc[mask_training].reset_index(drop=True)
test_raw = dataset_df.loc[mask_test].reset_index(drop=True)
valid_raw = dataset_df.loc[mask_valid].reset_index(drop=True)

X_train, y_train = process_csv_file(train_raw)
X_test, y_test = process_csv_file(test_raw)
X_valid, y_valid = process_csv_file(valid_raw)

model = Sequential()
model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same',
                        name='image_array', input_shape=(width,height,1)))
model.add(BatchNormalization())
model.add(Convolution2D(filters=16, kernel_size=(5, 5),
                        strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.25))

model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=32, kernel_size=(5, 5),
                        strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.25))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=64, kernel_size=(3, 3),
                        strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.25))

model.add(Convolution2D(filters=64, kernel_size=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                        strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(.25))

model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                        strides=(2, 2), padding='same'))

model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=num_labels, kernel_size=(3, 3),
                        strides=(2, 2), padding='same'))

model.add(Flatten())
model.add(Activation("softmax"))
model.summary()

# # Create the model
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,1)))
# # model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# # model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# # # model.add(BatchNormalization())
# model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
# # model.add(BatchNormalization())

# model.add(Flatten())
# model.add(Activation("softmax"))
# model.summary()

# model = Sequential()

# model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
# model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))

# model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))

# model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))

# model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.5))

# model.add(Flatten())

# model.add(Dense(2*2*2*num_features, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(2*2*num_features, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(2*num_features, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(num_labels, activation='softmax'))

# # Mostra o sumario da rede
# model.summary()

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(MODELPATH, monitor='val_loss', verbose=1, save_best_only=True)

model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_test), np.array(y_test)),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, checkpointer])


scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))