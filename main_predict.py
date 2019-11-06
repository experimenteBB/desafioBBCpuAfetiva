#Fazendo previsões
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('dataset_train',
                                                 target_size = (48, 48),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

def main(imagem_informada):
    test_image = image.load_img(imagem_informada, target_size = (48, 48))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    training_set.class_indices
    if result[0][0] == 0:
        prediction = 'Neutra'
    if result[0][0] == 1:
        prediction = 'Feliz'
    if result[0][0] == 2:
        prediction = 'Triste'
    if result[0][0] == 3:
        prediction = 'Surpreso'
    if result[0][0] == 4:
        prediction = 'Bravo'
    return prediction, result[0][0]

#dataset_test/s003-00_img.bmp
imagem_input = input('Informe o caminho da imagem:')
emocao_detectada = main(imagem_input)
emocao_detectada