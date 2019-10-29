import os.path
import os
import cv2
import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt


#Main script containing the image loading part. In keras-nn-example.py an example of a CNN using images loaded in that way is given.

####################Root directory containing the images to classify, folder are used as labels
# train --> cat --> img1, img2...imgN
#       --> dog --> img1, img2...imgN
#
# test  --> cat --> img1, img2...imgN
#       --> dog --> img1, img2...imgN

#sub folders in the root directory respresen classes, for example '/Users/michael/polyps' can contain a /neg/images... and pos/images...
root_dir_train = '/home/orlando/Documentos/desafioBBCpuAfetiva/dataset'

#Reshaping the images durring the loading process
image_w, image_h = 48,48

EMOTIONS = {0: "neutra" , 1: "feliz", 2: "triste", 3: "surpresa", 4: "brava"}

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
                result= re.search('-(.*)_', file)
                current_image_class = int(result.group(1))
                img = cv2.imread (imagepath)
                img = cv2.resize(img, (image_w, image_h))
                X_data.append(np.asarray(img, dtype="float32") / 255.0)
                Y_data.append(current_image_class)

    return np.array(X_data), np.array(Y_data)

from sklearn.model_selection import train_test_split

x_train, y_train = getimagedataandlabels(root_dir_train,image_w,image_h)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state = 0)

#Brining the labels into the correct format for categorical crossentropy
encoder = LabelBinarizer()
Y_train = encoder.fit_transform(y_train)
Y_test = encoder.fit_transform(y_test)

#A simple CNN model
model = Sequential()
model.add(Convolution2D(64, (3, 3),  activation="relu", input_shape=(image_w, image_h , 3)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
num_epochs = 100

model.fit(x_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)

haar_cascade_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
frame = cv2.imread('dataset/s001/bmp/s001-01_img.bmp')
image_copy = frame.copy()
gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
faces = haar_cascade_face.detectMultiScale(gray_image, scaleFactor=2.8, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

if len(faces) > 0:
    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    cv2.rectangle(image_copy, (fX, fY), (fX+fW, fY+fH), (0, 255, 0), 2)
    plt.imshow(image_copy)
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (image_w, image_h))
    roi = roi.astype("float32") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = model.predict(roi)

print(preds)
print(EMOTIONS[preds.argmax()])