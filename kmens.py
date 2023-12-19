import cv2

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