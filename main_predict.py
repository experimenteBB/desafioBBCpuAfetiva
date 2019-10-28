import glob, os, sys
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

# Colocar dentro de '' a chave enviada por e-mail
KEY = ''

# Colocar dentro de '' o endpoint enviado por e-mail
ENDPOINT = ''

# Autenticando com FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

def main(photo):
    IMAGES_FOLDER = os.path.join(os.path.abspath(''))

    # Obtendo a imagem para ser testada
    test_image_array = glob.glob(os.path.join(IMAGES_FOLDER, photo))
    image = open(test_image_array[0], 'r+b')

    # Identificando emoção
    detected_faces = face_client.face.detect_with_stream(image, return_face_attributes=['emotion'])

    if not detected_faces:
        raise Exception('No face detected from image {}'.format(photo))

    emotions = detected_faces[0].face_attributes.emotion
    notas_emotions = [emotions.neutral, emotions.happiness, emotions.sadness, emotions.surprise, emotions.anger]
    print (notas_emotions.index(max(notas_emotions)))

main(sys.argv[1:][0])