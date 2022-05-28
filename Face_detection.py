import cv2 as cv

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

img1 = cv.imread("src/pibes_1.jpg")
img2 = cv.imread("src/pibes_2.jpg")

down_width = 756
down_height = 1008
down_points = (down_width, down_height)
resized_down_img1 = cv.resize(img1, down_points, interpolation=cv.INTER_LINEAR)
resized_down_img2 = cv.resize(img2, down_points, interpolation=cv.INTER_LINEAR)

imgGray1 = cv.cvtColor(resized_down_img1, cv.COLOR_BGR2GRAY)
imgGray2 = cv.cvtColor(resized_down_img2, cv.COLOR_BGR2GRAY)

faces1 = face_cascade.detectMultiScale(imgGray1, 1.1, 4)
faces2 = face_cascade.detectMultiScale(imgGray2, 1.1, 4)

for (x, y, w, h) in faces1:
    cv.rectangle(resized_down_img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

for (x, y, w, h) in faces2:
    cv.rectangle(resized_down_img2, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv.imshow("Image1", resized_down_img1)
cv.imshow("Image2", resized_down_img2)
cv.waitKey(0)
