#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:37:48 2019

@author: orlando
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = plt.imread(filename)
	# plot the image
	plt.imshow(data)
	# get the context for drawing boxes
	ax = plt.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = cv2.Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
	# show the plot
	plt.show()
 
def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_faces(cascade, test_image, scaleFactor = 2.8):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image_copy


img_path = 's001-01_img.bmp'

haar_cascade_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
#faces_rects = haar_cascade_face.detectMultiScale(test_image_gray, scaleFactor = 2.8, minNeighbors = 5)

pixels = cv2.imread(img_path)

plt.imshow(detect_faces(haar_cascade_face, pixels))
cv2.destroyAllWindows()

emotion_classifier = load_model('models/_mini_XCEPTION.102-0.66.hdf5')
EMOTIONS = {"neutral": 0, "happy": 1, "sad": 2, "surprised": 3, "angry": 4}

orig_frame = cv2.imread(img_path)
frame = cv2.imread(img_path, 0)

plt.imshow(orig_frame)

#call the function to detect faces
#faces = detect_faces(haar_cascade_face, test_image2)
faces = haar_cascade_face.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

def predict_image(faces):
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

#cv2.imshow('test_face', orig_frame)
#cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
#cv2.destroyAllWindows()


#plt.imshow(convertToRGB(faces))