import cv2
import numpy as np
from zipfile import ZipFile
from PIL import Image # $ pip install pillow

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

with ZipFile('dataset/set_train.zip') as file:
   #print(file.namelist())
   for image in file.namelist():
      if(".bmp" in image):
         image_file = file.read(image)
         img = cv2.imdecode(np.frombuffer(image_file, np.uint8), 1)  
        
         # Convert into grayscale
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         # Detect faces
         faces = face_cascade.detectMultiScale(gray, 1.1, 4)
         # Draw rectangle around the faces
         for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

         # Display the output
         cv2.imshow('img', img)
         cv2.waitKey()
