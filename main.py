import cv2 as cv
import numpy as np


# Le a imagem em escala cinza
img = cv.imread('resources/eye1.bmp', 0)

# Aplica a equalização de histograma
equalize = cv.equalizeHist(img)

# Aplica filtro Gaussiano
blur = cv.GaussianBlur(img,(5,5),0)

# Converte a imagem para RGB
img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

# Binarização
_, binaryPupil = cv.threshold(blur, 30, 255, cv.THRESH_BINARY_INV)
binaryIris = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV,
21, 5)
cv.imwrite('output/binaryPupil.png', binaryPupil)
cv.imwrite('output/binaryIris.png', binaryIris)

# Fechamento da Imagem
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(13,13))
closingPupil = cv.morphologyEx(binaryPupil, cv.MORPH_CLOSE, kernel)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
closingIris = cv.morphologyEx(binaryIris, cv.MORPH_CLOSE, kernel)
cv.imwrite('output/closingPupil.png', closingPupil)
cv.imwrite('output/closingIris.png', closingIris)

# Encontrar o contorno na pupila
contoursPupil, _ = cv.findContours(closingPupil.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
areasPupil = [cv.arcLength(c,True) for c in contoursPupil]
max_index = np.argmax(areasPupil)

# Selecionar o maior dos contornos
cntPupil = contoursPupil[max_index]

# Encaixa uma elipse no contorno encontrado
ellipsePupil = cv.fitEllipse(cntPupil)
# Desenha a elipse na imagem
result = cv.ellipse(img.copy(), ellipsePupil ,(255,255,0), 1)
# Desenha o centro da elipse na imagem
cv.circle(result,(int(ellipsePupil[0][0]),int(ellipsePupil[0][1])),2,(0,255,0),1)

# Cria um círculo no centro do contorno para tirar as medidas da máscara
(x,y), radius = cv.minEnclosingCircle(cntPupil)
Pcenter = (int(x),int(y))
Pradius = int(radius)

# Criando uma máscara para detectar a iris
mask = np.zeros_like(closingIris)
mask = cv.circle(mask, Pcenter, Pradius * 3, (255,255,255), -1)

# Recortando a área da Iris
irisMask = cv.bitwise_and(closingIris, mask)
cv.imwrite('output/IrisMask.png', irisMask)

# # Abertura da máscara
# kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
# irisMask = cv.morphologyEx(irisMask, cv.MORPH_OPEN, kernel)
# cv.imwrite('output/IrisMaskOpen.png', irisMask)

# # Fechamento da máscara
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
# irisMask = cv.morphologyEx(irisMask, cv.MORPH_CLOSE, kernel)
# cv.imwrite('output/IrisMaskClose.png', irisMask)

# Encontrar o contorno na iris
contoursIris, _ = cv.findContours(irisMask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
areasIris = [cv.arcLength(c,True) for c in contoursIris]
max_index2 = np.argmax(areasIris)

# Selecionar o maior dos contornos
cntIris = contoursIris[max_index2]

# Encaixa uma elipse no contorno encontrado
ellipseIris = cv.fitEllipse(cntIris)
# Desenha a elipse na imagem
result = cv.ellipse(result.copy(), ellipseIris,(255,255,0), 1)


cv.imwrite('output/Iris.png', result)