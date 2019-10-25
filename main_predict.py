from joblib import dump, load
from imutils import face_utils
import dlib
import cv2
import numpy as np
from PIL import Image
import os
import sys

from joblib import dump, load

##############
#USO:
#python main_predict.py caminhodoArquivo.jpg

#Modelo treinado para uso deste exemplo:
#LogReg_BBFace.joblib

#OBS: É Necessario baixar na mesma pasta o modelo treinado shape_predictor_68_face_landmarks.dat disponivel em 
#https://github.com/italojs/facial-landmarks-recognition-/blob/master/shape_predictor_68_face_landmarks.dat


############
if __name__ == '__main__':

    clf = load('LogReg_BBFace.joblib') 
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    sentimentosDict = {
            '0' : 'neutro',
            '1' : 'feliz',
            '2' : 'triste',
            '3' : 'surpreso',
            '4' : 'bravo'
            }
    
    imagem = sys.argv[1]
    imagemFace = cv2.imread(imagem)
    gray = cv2.cvtColor(imagemFace, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    faceDetect = shape.reshape(-1)
    #faceDetect.shape
    face = faceDetect.reshape(1,-1)
    #sentimentosDict[str(clf.predict(SingleSample)[0])]
    predicted= str(clf.predict(face)[0])
    print(predicted+"-"+ sentimentosDict[predicted])

    exit()





