import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import align_dlib as opf
from tensorflow import keras

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=False,
	help="image path")
args = vars(ap.parse_args())

model = tf.keras.models.load_model('models/cnn_model.hdf5')
align = opf.AlignDlib('models/shape_predictor_68_face_landmarks.dat')

def main():
    file = args["file"]
    data = image_prep(file)
    prediction = predict(data)
    print(prediction[0])
    return(prediction[0])

def image_prep(file):
    img = cv2.imread(file)
    bb = align.getLargestFaceBoundingBox(img)
    alignedFace = align.align(256, img, bb, landmarkIndices=opf.AlignDlib.OUTER_EYES_AND_NOSE)
    data =[]
    data.append(np.asarray(alignedFace, dtype="int32"))
    data = np.array(data)
    data = data / 256

    return data

def ohe_dec(data):
    tam = len(data)
    p = 0
    lst = []
    if p < tam:
        ohe = np.argmax(data.iloc[p])
        lst.append(ohe)
        p = p+1

    return(lst)

def predict(data):
    pred = model.predict(data)
    pred = pd.DataFrame(pred)
    pred = ohe_dec(pred)

    return pred

def fer(path):
    file = path
    data = image_prep(file)
    prediction = predict(data)
    return(prediction[0])

if __name__ == '__main__':
    main()
