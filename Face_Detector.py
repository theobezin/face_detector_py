import cv2

# charger les données entrainées à reconnaitre un visage
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# importer l'image
img = cv2.imread('IMG_TEST.jpg')


# convertir en gris
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detecter le visage
#face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# afficher l'image
cv2.imshow('Face Detector Sadeuh', grayscaled_img)

# à mettre avant la fin pour terminer le programme au click
cv2.waitKey()

print("Code completed")
