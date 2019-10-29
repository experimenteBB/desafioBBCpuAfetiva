import cv2
import glob
import random
import dlib
import sys
import numpy as np
from sklearn.svm import SVC

emocoes = ["neutra", "feliz", "triste", "surpreso", "bravo"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
preditor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = SVC(kernel='linear', probability=True, tol=1e-3)

def get_arquivos(emocao):
    arquivos = glob.glob("dataset//%s//*" %emocao)
    random.shuffle(arquivos)
    treino = arquivos[:int(len(arquivos))] 
    arquivo_predicao = [sys.argv[1]]
    return treino, arquivo_predicao

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections):
        shape = preditor(image, d) 
        xlist = []
        ylist = []
        for i in range(1,68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
 
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)

    if len(detections) < 1: 
        landmarks_vectorised = "error"
    return landmarks_vectorised

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    training = []
    prediction = []
    for emocao in emocoes:
        training, prediction = get_arquivos(emocao)
        for item in training:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                training_data.append(landmarks_vectorised)
                training_labels.append(emocoes.index(emocao))
 
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            if landmarks_vectorised == "error":
                pass
            else:
                prediction_data.append(landmarks_vectorised)
                prediction_labels.append(emocoes.index(emocao))

    return training_data, training_labels, prediction_data, prediction_labels

def main():
    accur_lin = []
    limite_execucoes = 3

    for i in range(0,limite_execucoes):
        print("Iteracao %s/%s" % (i + 1, limite_execucoes))
        print("Construindo set...")
        training_data, training_labels, prediction_data, prediction_labels = make_sets()

        npar_train = np.array(training_data)
        npar_trainlabs = np.array(training_labels)
        print("Treinando SVM linear...")
        clf.fit(npar_train, training_labels)

        print("Obtendo precisao...")
        npar_pred = np.array(prediction_data)
        pred_lin = clf.score(npar_pred, prediction_labels)
        accur_lin.append(pred_lin)
        probabilidade = clf.predict_proba(prediction_data)
        print("")

    indice_valor_maximo = np.argmax(probabilidade[0,:])

    print("%s-%s" % (indice_valor_maximo, emocoes[indice_valor_maximo]))

if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print("\n===============================================================")
        print("USO: python main_predict.py </caminho/do/arquivo_de_imagem.???>")
        print("===============================================================\n")
        exit(0)
    main()
