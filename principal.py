from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2

#alterar o caminho da imagem nas linhas 11, 18 e 33

image1 = Image.open("repositorio/s001-01_img.bmp")
image_array1 = np.array(image1)
# plt.imshow(image_array1)  # Retirar o comentário inicial para mostrar a imagem

plt.imshow(image1)
plt.show()

image = face_recognition.load_image_file("repositorio/s001-01_img.bmp")
face_locations = face_recognition.face_locations(image)
print(face_locations)

top, right, bottom, left = face_locations[0]
face_image1 = image[top:bottom, left:right]
plt.imshow(face_image1)
image_save = Image.fromarray(face_image1)
image_save.save("imagem_saida.jpg")

#Detectando emoção

emotion_dict = {'Bravo': 0, 'Triste': 5, 'Neutro': 4, 'Sem identificacao': 1, 'Surpreso': 6, 'Nao identificado': 2, 'Feliz': 3}

#
face_image = cv2.imread("repositorio/s001-01_img.bmp")
plt.imshow(face_image)
# plt.imshow(face_image)  # Retirar o comentário inicial para mostrar a imagem
print(face_image.shape)

# resizing the image
face_image = cv2.resize(face_image, (48,48))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

#Carregando o modelo treinado:
#-----------------------------

model = load_model("simple_CNN.530-0.65.hdf5")
print(face_image.shape)

predicted_class = np.argmax(model.predict(face_image))

label_map = dict((v, k) for k, v in emotion_dict.items())
predicted_label = label_map[predicted_class]
print(predicted_label)

