import cv2 as cv
import numpy as np

img = cv.imread("src/hoja.jpeg")  # Define las imagenes a inspeccionar

imgRezised = cv.resize(img, (700, 700))  # Tamaño de la imagen original


width, height = 700, 250  # Define el tamaño que va a tener la imagen en perspectiva

width, height = 700, 250
pts1 = np.float32([[72, 275], [620, 236], [77, 431], [683, 372]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv.getPerspectiveTransform(pts1, pts2)
imgOutput = cv.warpPerspective(imgRezised, matrix, (width, height))


pts1 = np.float32([[72, 275], [620, 236], [77, 431], [683, 372]])  # Puntos de los vertices en la imagen original

pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # Puntos de los vertices en la imagen en perspectiva

matrix = cv.getPerspectiveTransform(pts1, pts2)  # Transforma la imagen entre los puntos de la original a los de la imagen en perspectiva

imgOutput = cv.warpPerspective(imgRezised, matrix, (width, height))  # Crea la imagen en perspectiva

cv.imshow("OriginalImage", imgRezised)  # Muestra las imagenes
cv.imshow("Image", imgOutput)

cv.waitKey(0)