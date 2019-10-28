import os.path
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.preprocessing import LabelBinarizer



####################Root directory containing the images to classify, folder are used as labels
# train --> cat --> img1, img2...imgN
#       --> dog --> img1, img2...imgN
#
# test  --> cat --> img1, img2...imgN
#       --> dog --> img1, img2...imgN

#Here the training and test folders can be defined
#sub folders in the root directory respresen classes, for example '/Users/michael/polyps' can contain a /neg/images... and pos/images...
root_dir_train = '/Users/michael/winterfell/ml-stuff/data/train'
root_dir_test = '/Users/michael/winterfell/ml-stuff/data/test'

#Reshaping the images durring the loading process
image_w, image_h = 256,256

def getimagedataandlabels(root_dir, image_w, image_h):

    X_data=[]
    Y_data=[]
    classes_from_directories = []  # to determine the classes from the root folder structure automatically

    for directory, subdirectories, files in os.walk(root_dir):
        # print(directory)
        for subdirectory in subdirectories:
            # print(subdirectory)
            classes_from_directories.append(subdirectory)
        for file in files:
            # print(file)
            # print(directory)
            if file != '.DS_Store':  # fix for MAC...
                imagepath = os.path.join(directory, file)
                current_image_class_splitt = imagepath.split('/')
                current_image_class = current_image_class_splitt[len(current_image_class_splitt) - 2]
                img = cv2.imread (imagepath)
                img = cv2.resize(img, (image_w, image_h))
                X_data.append(np.asarray(img, dtype="int32"))
                Y_data.append(current_image_class)
                #print imagepath

    return np.array(X_data), np.array(Y_data)

#Loading the training images
x_train, y_train = getimagedataandlabels(root_dir_train,image_w,image_h)
print "Training images and labels loaded"
print x_train.shape
print y_train.shape

#Loading the te
x_test, y_test = getimagedataandlabels(root_dir_test,image_w,image_h)
print "Test images and labels loaded"
print x_test.shape
print y_test.shape


#Brining the labels into the correct format for categorical crossentropy
encoder = LabelBinarizer()
Y_train = encoder.fit_transform(y_train)
Y_test = encoder.fit_transform(y_test)


#A simple CNN model
model = Sequential()
model.add(Convolution2D(32, (3, 3),  activation="relu", input_shape=(256, 256 , 3)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, Y_train,
          batch_size=2, epochd=10, verbose=1)

evaluation = model.evaluate(x_test, Y_test, verbose=0)
print evaluation
