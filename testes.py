#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:45:24 2019

@author: orlando
"""

import cv2,os
import numpy as np
from PIL import Image

#file = open('s001-00_img.bmp', 'r', encoding='latin-1')
pilImage=Image.open('s001-00_img.bmp').convert('L')
imageNp=np.array(pilImage,'uint8')
faceSamples=[]
detector= cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml");
faces=detector.detectMultiScale(imageNp)
#If a face is there then append that in the list as well as Id of it
for (x,y,w,h) in faces:
    faceSamples.append(imageNp[y:y+h,x:x+w])
#img = img_as_float(file)
#emotion = int(float(file.readline()))


#recognizer = cv2.createLBPHFaceRecognizer()


def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('dataset/s001/bmp')
#recognizer.train(faces, np.array(Ids))
#recognizer.save('trainner/trainner.yml')