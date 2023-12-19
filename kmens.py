import cv2
import numpy as np
import warnings
from sklearn.cluster import KMeans
import os

# Suprimir avisos específicos do joblib
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Lista de imagens
lista_imagens = [
    'Cachorro_0.png',
    'Cachorro_1.png',
    'Cachorro_2.png',
    'Cachorro_3.png',
    'Cachorro_4.png',
    'Cachorro_5.png'
]

# Número de clusters
clusters = 20

for imagem in lista_imagens:
    # Leitura da imagem
    image = cv2.imread(imagem)
    # Transforma a imagem em unidimensional
    unidimensional = image.reshape(-1, 3)
    
    # Aplicando K-means
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    kmeans.fit(unidimensional)