import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('DADOS/haarcascades/haarcascade_frontalface_default.xml') # xml haarcascade face frontal

imagem_entrada = cv2.imread('s002-03_img.bmp') # arquivo a ser predito

class ExpressaoFacial:
    def __init__(self, imagem_carregada):
        self.imagem_carregada = imagem_carregada
        print("Objeto criado")
    
    def detectar_roi(self, imagem_carregada):
        face_img = cv2.cvtColor(imagem_carregada, cv2.COLOR_BGR2GRAY) # converte a imagem de BGR (padrão OCV) para tons de cinza
        face = face_cascade.detectMultiScale(face_img)
        cie  = face[0][0]        # canto inferior esquerdo
        cse  = face[0][1]        # canto superior esquerdo
        cid  = cie + face[0][2]  # canto inferior direito
        csd  = cse + face[0][3]  # canto superior direito

        roi_img = face_img[cse:csd, cie:cid]
        img = cv2.resize(roi_img,(150, 150), interpolation = cv2.INTER_CUBIC)
        return img

    def retorna_previsao(self):
        imagem = self.detectar_roi(self.imagem_carregada)
        imagem = np.array(imagem)
        imagem = imagem.reshape(1, 150, 150, 1)
    
        model = load_model('BBi9FaceSentimentos.h5')
    
        y_pred = model.predict_classes(imagem)
        return y_pred

def main(imagem_carregada):
    expressao_facial = ExpressaoFacial(imagem_entrada)
    previsao = expressao_facial.retorna_previsao()
    print(previsao[0])


if __name__ == '__main__':
    main(imagem_entrada)