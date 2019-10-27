import cv2
import argparse
import math
import numpy as np
import dlib
import sys
from sklearn.externals import joblib

def main():
    
    SUPPORT_VECTOR_MACHINE_clf2 = joblib.load('aps_modelo.pkl')
    emocoes = ["Neutra", "Feliz", "Triste", "Surpresa", "Brava"]
    img = cv2.imread(args["image"])
    treated_img = treat_face(img)
    landmarks_vec = get_landmarks(treated_img)
    pred_data = []
    pred_data.append(landmarks_vec)
    a = SUPPORT_VECTOR_MACHINE_clf2.predict(pred_data) 
    print('EXPRESSAO FACIAL DETECTADA:', emocoes[a[0]])

def treat_face(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    # https://github.com/opencv/opencv/tree/master/data/haarcascades
    face1 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
    face2 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
    face3 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
    face4 = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt_tree.xml")
    face_1 = face1.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_2 = face2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_3 = face3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),flags=cv2.CASCADE_SCALE_IMAGE)
    face_4 = face4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face_1)==1:
        req_face=face_1
    elif len(face_2) == 1:
        req_face = face_2
    elif len(face_3) == 1:
        req_face = face_3
    elif len(face_4) == 1:
        req_face = face_4
    else:
        req_face=""
    
    if len(req_face) == 1:
        for (x, y, w, h) in req_face:
            roi_gray = gray[y:y + h, x:x + w]
        
        cropped_img = cv2.resize(roi_gray, (350, 350))
        img_treated = clahe.apply(cropped_img)
        
        return img_treated 
    else:
        print("Nenhuma face foi encontrada na imagem!")
        sys.exit()

def get_landmarks(image_p):
    
    face_det = dlib.get_frontal_face_detector()
    land_pred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
    face_detections = face_det(image_p,1)
    
    for k,d in enumerate(face_detections):
        shape = land_pred(image_p,d)
        x_cords = []
        y_cords = []
        
        for i in range(1,68):
            x_cords.append(float(shape.part(i).x))
            y_cords.append(float(shape.part(i).y))

        xmean = np.mean(x_cords)
        ymean = np.mean(y_cords)
        x_central = [(x-xmean) for x in x_cords] 
        y_central = [(y-ymean) for y in y_cords]

        if x_cords[26] == x_cords[29]: 
            anglenose=0
        else:
            anglenose_rad = int(math.atan((y_central[26]-y_central[29])/(x_central[26]-x_central[29])))
            anglenose = int(math.degrees(anglenose_rad))

        if anglenose<0:
            anglenose += 90
        else:
            anglenose -= 90      

        landmarks_v = []
        
        for x,y,w,z in zip(x_central,y_central,x_cords,y_cords):
            landmarks_v.append(x) 
            landmarks_v.append(y) 
            np_mean_co = np.asarray((ymean,xmean))
            np_coor = np.asarray((z,w))
            euclid_d = np.linalg.norm(np_coor-np_mean_co)
            landmarks_v.append(euclid_d)
            angle_rad = (math.atan((z-ymean)/(w-xmean)))
            angle_degree = math.degrees(angle_rad)
            angle_req = int(angle_degree-anglenose)
            landmarks_v.append(angle_req)

    return landmarks_v

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--image", required = True, help='img path')
    args = vars(parser.parse_args())
    main()