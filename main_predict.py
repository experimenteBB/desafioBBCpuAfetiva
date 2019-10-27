from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import sys
from joblib import dump, load
import numpy as np

from imutils import face_utils
import dlib
import cv2
 
 
def main():
 	# p = modelo pré-treinado para reconhecer 68 pontos da face
	p = "./lib/shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(p)

	# arquivo='s016-03_img.bmp'


	arquivo = sys.argv[1]

	image = cv2.imread(arquivo)
	# image = cv2.imread('time.jpg')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 0)

	# Make the prediction and transfom it to numpy array
	shape = predictor(gray, rects[0])
	shape = face_utils.shape_to_np(shape)

	shape_transformado = [shape.reshape(-1)]
	#shape_transformado = shape_transformado.reshape(1,-1)

	#print(shape_transformado)

	modelo=load('./resultado/MLPGridSearch69acc.joblib')

	sc = load('./resultado/sc.joblib')
	# sc = StandardScaler()
	#sc.fit(shape_transformado)

	imagens_treino_pad = sc.transform(shape_transformado)


	#print(imagens_treino_pad)
	resultado = modelo.predict(imagens_treino_pad)
	print(resultado[0])
	
main()