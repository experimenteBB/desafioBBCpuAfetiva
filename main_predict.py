import sys
import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
emotion_dict = {0: "neutra", 1: "feliz", 2: "triste", 3: "surpreso", 4: "bravo"}
MODELPATH = "./models/model.h5"
model = load_model(MODELPATH)

if __name__ == '__main__':

   if(len(sys.argv) < 2):
      print("Por favor informe o caminho de uma imagem...")
      sys.exit()
   else:
      print("Detectando emoção na image...")
      img = cv2.imread(str(sys.argv[1]))

      # Convert into grayscale
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # Detect faces
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)

      # Draw rectangle around the faces
      for (x, y, w, h) in faces:
         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

         roi_gray = gray[y:y + h, x:x + w]
         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
         # cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
         prediction = model.predict(cropped_img)
         #cv2.putText(img, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
         
         #Converte pro modo BB
         #0-neutra; 1-feliz; 2-triste; 3-surpreso; e 4-bravo.
         print(int(np.argmax(prediction)))
         

      # Display the output
      # cv2.imshow('img', img)
      # cv2.waitKey()
