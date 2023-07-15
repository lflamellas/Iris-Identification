import cv2 as cv
import numpy as np

# Ler a imagem em escala cinza
img = cv.imread('resources/eye.bmp', 0)


# Normalização Geométrica

# Normalização de Tonalidade
equalize = cv.equalizeHist(img)
cv.imwrite('output/equalize.png', equalize)

blur = cv.GaussianBlur(equalize, (7,7), 0)

# Detecção da Iris
edges = cv.Canny(blur, 50, 5, 5)
cv.imwrite('output/edges.png', edges)

kernel = np.ones([3,3])
close = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
cv.imwrite('output/close.png', close)

# Encontrar círculos 
circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=1, minDist=1, param1=100, param2=30, minRadius=0, maxRadius=80)

# Verificar se foram encontrados círculos
if circles is not None:
    # Converter as coordenadas e o raio do círculo para inteiros
    circles = np.round(circles[0, :]).astype(int)

    # Selecionar o círculo com maior raio (íris)
    idx = circles[np.argsort(circles[:, 2])]

    # Extrair as coordenadas e o raio da íris
    iris = idx[-1]
    pupil = idx[-2]

    print(iris)
    print(pupil)

    # Desenhar um círculo ao redor da iris na imagem original do olho
    cv.circle(equalize, (iris[0], iris[1]), iris[2], (0, 0, 0), 2)

    # Desenhar um círculo ao redor da pupila na imagem original do olho
    cv.circle(equalize, (pupil[0], pupil[1]), pupil[2], (0, 0, 0), 2)

    # Salvar a imagem com os círculos
    cv.imwrite('output/circles.png', equalize)


# Extração da Iris

# Identificação da Iris
