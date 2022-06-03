import cv2 as cv

# Carga la HAAR CASCADE de un rostro
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define las imagenes a inspeccionar
img1 = cv.imread("src/pibes_1.jpg")
img2 = cv.imread("src/pibes_2.jpg")

# Tamaño de la imagen final
down_width = 756
down_height = 1008
down_points = (down_width, down_height)

# Achicamos las imagenes ya que sino no entra en la pantalla por su resolucion tan alta
resized_down_img1 = cv.resize(img1, down_points, interpolation=cv.INTER_LINEAR)
resized_down_img2 = cv.resize(img2, down_points, interpolation=cv.INTER_LINEAR)

# Aplicamos un filtro de escala de grises para mejorar la efectividad
imgGray1 = cv.cvtColor(resized_down_img1, cv.COLOR_BGR2GRAY)
imgGray2 = cv.cvtColor(resized_down_img2, cv.COLOR_BGR2GRAY)

# Devuelve un rectangulo con coordenadas alrededor de los rostros detectados
faces1 = face_cascade.detectMultiScale(imgGray1, 1.1, 4)
faces2 = face_cascade.detectMultiScale(imgGray2, 1.1, 4)

# Dibuja los rectangulos arriba mencionados
for (x, y, w, h) in faces1:
    cv.rectangle(resized_down_img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

for (x, y, w, h) in faces2:
    cv.rectangle(resized_down_img2, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Muestra las imagenes
cv.imshow("Image1", resized_down_img1)
cv.imshow("Image2", resized_down_img2)
cv.waitKey(0)
