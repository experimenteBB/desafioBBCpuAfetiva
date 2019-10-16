# Desafio BB Experimente de Computação Afetiva

## Qual desafio
Desenvolver um algoritmo de análise de sentimentos a partir de imagens faciais de brasileiros utilizando Inteligência Artificial e Processamento de Imagens.

## Como acessar

## Aonde consigo materiais para me ajudar
É possível encontrar materiais de apoio no site da OpenCV (https://opencv.org/) e no site da Dlib (http://dlib.net).
No site da OpenCV pode-se encontrar algoritmos de detecção facial, por exemplo em https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html, ou em sites de terceiros como https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81, que serão necessários para detectar as faces nas imagens.
 
No site da dlib pode-se também detectar faces e pontos "especiais" (canto dos olhos, da boca, etc.) a fim de determinar se a pessoa está sorrindo, por exemplo (vide http://dlib.net/face_landmark_detection.py.html).   
 
Vale consultar o Google!
 
A base de imagens faciais de brasileiros para a configuração/treinamento de seu método está disponível na Plataforma Analítica. Cada pessoa da base possui uma pasta com cinco imagens, cada uma delas contendo uma imagem facial exibindo um dos 5 principais sentimentos a ser analisados: neutro, feliz, triste, surpreso e bravo. As imagens de cada indivíduo já estão nomeadas nesta ordem para facilitar sua identificação, por exemplo, a imagem "s001-00_img.bmp" ilustra a pessoa s001 com a face neutra; a imagem  "s001-01_img.bmp" ilustra a pessoa s001 com a face feliz; a "s001-02_img.bmp" com a face triste; a "s001-03_img.bmp" com a face surpresa; e a "s001-04_img.bmp" com a face brava.

Utilizem a imaginação e as bibliotecas disponibilizadas para extrair as melhores características e classificar o máximo de imagens corretamente.

Obs: A base de dados fornecida é exclusivamente para pesquisa. Não a utilize para fins comerciais.

## Aonde e como entregar

Ao final, juntamente com seu código, você deverá entregar, na Plataforma Analítica, um binário que receba uma imagem e gere uma saída inteira representando a emoção facial na imagem: 0-neutra; 1-feliz; 2-triste; 3-surpreso; e 4-bravo. Testaremos seu código e binário em outra bases de imagens faciais brasileiras com as mesmas expressões faciais para vermos quão robusto é seu método e divulgaremos o resultado também na Plataforma.
 
Bom trabalho!
