import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import pickle

emotions = ["00", "01", "02", "03", "04"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
clf = SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) 
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
#clf3 = GaussianNB()
#clf = VotingClassifier(estimators=[('SVC', clf1), ('RF', clf2), ('gnb', clf3)], voting='hard')

data = {} 
accur_lin = []

def get_files(emotion): 
    files = glob.glob('**/*%s_img.bmp' %emotion, recursive=True)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] 
    prediction = files[-int(len(files)*0.2):] 
    return training, prediction

def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): 
        shape = predictor(image, d) 
        xlist = []
        ylist = []
        for i in range(1,68): #
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist) 
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] 
        ycentral = [(y-ymean) for y in ylist]
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print("Emoção %s" %emotion)
        training, prediction = get_files(emotion)
        
        for item in training:
            image = cv2.imread(item) 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("Nenhuma face detectada")
            else:
                training_data.append(data['landmarks_vectorised']) 
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("Nenhum face detectada")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


def train():
    for i in range(0,5):
    print("Set %s" %i) 
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data) 
    npar_trainlabs = np.array(training_labels)
    print("Treinamento SVM linear %s" %i)
    clf.fit(npar_train, training_labels)
    print("Acuracia %s" %i) 
    npar_pred = np.array(prediction_data)
    pred_lin = clf.score(npar_pred, prediction_labels)
    print ("linear: ", pred_lin)
    accur_lin.append(pred_lin) 
    print("Média de acurácia lin svm: %s" %np.mean(accur_lin))
    pickle.dump(clf, open('modelo.sav', 'wb'))

def main_predict(image):

    #image = cv2.imread("teste2.bmp")
    clf = pickle.load(open('modelo.sav', 'rb'))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    clahe_image = clahe.apply(gray)
    get_landmarks(clahe_image)
    predict_data = []
    predict_data.append(data['landmarks_vectorised'])
    result = clf.predict(predict_data)
    return result[0]