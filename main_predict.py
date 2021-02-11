from pathlib import Path
import numpy as np
import pandas as pd
import dlib
import cv2
import sys
import os
import tensorflow as tf
import align_dlib as opf
from tensorflow import keras
import align_dlib as opf

#para baixar os arquivos necessários basta descomentar o código abaixo
#from urllib.request import urlretrieve

#def download(url, file):
#    if not os.path.isfile(file):
#        print("Download file... " + file + " ...")
#        urlretrieve(url,file)
#        print("File downloaded")
#        
#download('https://github.com/cmusatyalab/openface/raw/master/openface/align_dlib.py', 'align_dlib.py')
#download('https://github.com/JeffTrain/selfie/raw/master/shape_predictor_68_face_landmarks.dat', 'shape_predictor_68_face_landmarks.dat')
#print("All the files are downloaded")

image = cv2.imread('001-00_img.bmp')

if(len(sys.argv) < 2):
  print("Por favor informe o caminho de uma imagem...")
  sys.exit()
else:
  image = cv2.imread(str(sys.argv[1]))


model = tf.keras.models.load_model('model.hdf5')
align = opf.AlignDlib('shape_predictor_68_face_landmarks.dat')

#preprocessamento da imagem
def imgpreprocessing():
    bb = align.getLargestFaceBoundingBox(image)
    alignedFace = align.align(256, image, bb, landmarkIndices=opf.AlignDlib.OUTER_EYES_AND_NOSE)
    img = tf.keras.preprocessing.image.img_to_array(alignedFace)
    img = tf.keras.applications.mobilenet.preprocess_input(img[tf.newaxis,...])
    
    return img

def main():
    img = imgpreprocessing()
    res = model.predict(img)
    m = max(res[0])
    p = -1
    for i, val in enumerate(res[0]):
        if val == m:
            p = i
    print('===============> '+ str(p))
    return p

if __name__ == '__main__':
    main()