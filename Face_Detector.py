import cv2

# charger les données entrainées à reconnaitre un visage
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

#AVEC UNE IMAGE
"""
# importer l'image
img = cv2.imread('IMG_TEST_3.jpg')

# convertir en gris
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detecter le visage
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# print(face_coordinates)

# dessiner un rectangle avec les coordonnées
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# afficher l'image
cv2.imshow('Face Detector Sadeuh', img)

# à mettre avant la fin pour terminer le programme au click
cv2.waitKey()
"""

#AVEC UNE VIDEO
#capturer la webcam
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detector Sadeuh', frame)

    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()





print("Code completed")
