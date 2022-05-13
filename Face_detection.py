import cv2 as cv

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

img1 = cv.imread("src/pibes_1.jpg")

down_width = 756
down_height = 1008
down_points = (down_width, down_height)
resized_down_img1 = cv.resize(img1, down_points, interpolation=cv.INTER_LINEAR)

imgGray1 = cv.cvtColor(resized_down_img1, cv.COLOR_BGR2GRAY)

faces1 = face_cascade.detectMultiScale(imgGray1, 1.1, 4)

for (x, y, w, h) in faces1:
    cv.rectangle(resized_down_img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv.imshow("Image", resized_down_img1)
cv.waitKey(0)
