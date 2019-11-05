#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:37:48 2019

@author: orlando
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt



def predict(img_path, showImage=False):
    
    image_w, image_h = 48, 48
    
    # parameters for loading data and images
    detection_model_path = 'haarcascade/haarcascade_frontalface_default.xml'
    model = load_model('models/best_model.hdf5', compile=False)
    
    #0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
    
    EMOTIONS = {0: "neutra" , 1: "feliz", 2: "triste", 3: "surpresa", 4: "bravo"}
    TRANSLATE = {0: 4, 1: 1, 2: 2, 3: 3, 4: 0}
    
    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
     
    #reading the frame
    orig_frame = cv2.imread(img_path)
    frame = cv2.imread(img_path,0)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (image_w, image_h))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi)[0]
        sentiment = EMOTIONS[TRANSLATE[preds.argmax()]]
        if showImage:
            cv2.putText(orig_frame, sentiment, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
            cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(255, 0, 0), 2)
            plt.imshow(orig_frame)
    
    return TRANSLATE[preds.argmax()]

def main():
    #image = 'images.jpeg'
    image = sys.argv[1]
    print(predict(image, False))
    
if __name__ == '__main__':
    main()
