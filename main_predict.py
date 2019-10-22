import cv2
import os
import numpy as np
from PIL import Image
import sys

##############
#USO:
#python predictMain.py caminhodoArquivo.jpg


############
if __name__ == '__main__':

    detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read("classificadorSentimentosMLPHv3.yml")
    largura, altura = 640, 480

    sentimentosDict = {
        '0' : 'neutro',
        '1' : 'feliz',
        '2' : 'triste',
        '3' : 'surpreso',
        '4' : 'bravo'
    }

    #print(sys.argv[1])

    imagem = sys.argv[1]
    imagemFace = Image.open(imagem).convert('L')
    imagemFaceNP = np.array(imagemFace, 'uint8')
    facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemFaceNP[y:y+a,x:x +l], (largura,altura))
        imagemFaceNP2 = np.array(imagemFace, 'uint8')
        idprevisto, confianca = reconhecedor.predict(imagemFaceNP2)
        print(str(idprevisto) + "-" + sentimentosDict[str(idprevisto)]) 
    exit()





