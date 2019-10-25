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














    #print("modelo carregado...")
    detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def trataImage(image, target_size):
        #plt.imshow(image)
        
        #image = image.resize(target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        #image.reshape(1,680,480,3)
        #image = np.array(image)
  #     image = np.array(image, dtype="float") / 255.0
   #    image = np.expand_dims(image,axis=0)

        return image


# def preprocess_image(image, target_size):
#     if image.mode != "RGB":
#         image = image.convert("RGB")
#     image = image.resize(target_size)
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)

#     return image


    def classifica(imagem):
        imagetrat = trataImage(imagem,  target_size=(640, 480))
        sentimentosDict = {
            '0' : 'neutro',
            '1' : 'feliz',
            '2' : 'triste',
            '3' : 'surpreso',
            '4' : 'bravo'
            }
        
        clf = load('LogReg_BBFace.joblib') 
        
        new_model = tf.keras.models.load_model('saved_models/keras_cifar10_trained_modelBBFaces.h5')
        lista = new_model.predict(imagetrat)
    
        pos = maior = menor = indice = indicemai = indicemen = 0
        while pos < len(lista):
            if pos == 0:
                maior = menor = lista[pos]
                indice = pos
            else:
                if lista[pos] > maior:
                    maior = lista[pos]
                    indicemai = pos
            pos += 1

        print(str(indicemai)+"-"+ sentimentosDict[str(indicemai)])
        print(str(lista))
        #
        # 
        # 
        # return(str(indicemai)+"-"+ sentimentosDict[str(indicemai)])

    

    largura, altura = 480, 640
  
    imagem = sys.argv[1]
    #imagemFace = Image.open(imagem).convert('L')
    #if imagemFace.mode != "RGB":
    #    imagemFace = imagemFace.convert("RGB")
    img = cv2.imread (imagem)
    img = cv2.resize(img, (largura, altura))
    #imagemFaceNP = np.array(img, 'uint8')
    #facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)

    #for (x, y, l, a) in facesDetectadas:
    #imagemFace = cv2.resize(imagemFaceNP[y:y+a,x:x +l], (largura,altura))
    #imagemFaceNP2 = np.array(imagemFace, 'uint8')
    
    idprevisto = classifica(img)
        #print(str(idprevisto) + "-" + sentimentosDict[str(idprevisto)]) 
    exit()





