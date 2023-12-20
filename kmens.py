import cv2
import numpy as np
import warnings
from sklearn.cluster import KMeans
import os

# Suprimir avisos específicos do joblib
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Lista de imagens
lista_imagens = [
    #'Cachorro_new_0.png'#,
    #'Cachorro_new_1.png',
    #'Cachorro_new_2.png',
    #'Cachorro_new_3.png',
    #'Cachorro_new_4.png',
    'Cachorro_new_5.png'
]

# Número de clusters
clusters = 1

for imagem in lista_imagens:
    # Leitura da imagem
    image = cv2.imread(imagem)
    # Transforma a imagem em unidimensional
    unidimensional = image.reshape(-1, 3)
    
    # Aplicando K-means
    kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=42)
    kmeans.fit(unidimensional)

    # Obtém os rótulos de cluster e os centros dos clusters para reconstruir a imagem segmentada
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape)
    
    # Escreve a imagem segmentada em um novo arquivo
    cv2.imwrite(f'Imagem_Segmentada_{clusters}_{imagem}', segmented_image)
    
    print('------------------------------------------------------------')
    print(f'Cores na imagem antes do processamento:', len(np.unique(image.reshape(-1, image.shape[2]), axis=0)))
    print(f'Cores na imagem:', len(np.unique(segmented_image)))  # retorna quantidade de cores diferentes

    file_size = os.path.getsize(f'Imagem_Segmentada_{clusters}_{imagem}')
    print("O tamanho do arquivo :", file_size/1000, "Kbytes")
