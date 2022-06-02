import cv2 as cv
import numpy as np

img = cv.imread("src/hoja.jpeg")

imgRezised = cv.resize(img, (700, 700))

width, height = 700, 250
pts1 = np.float32([[72, 275], [620, 236], [77, 431], [683, 372]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv.getPerspectiveTransform(pts1, pts2)
imgOutput = cv.warpPerspective(imgRezised, matrix, (width, height))

cv.imshow("OriginalImage", imgRezised)
cv.imshow("Image", imgOutput)

cv.waitKey(0)
