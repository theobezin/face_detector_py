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

    #lecture de chaque frame du flux de la webcam
    successful_frame_read, frame = webcam.read()

    #conversion en nuance de gris
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detection des coordonnées du visage grâce aux données entrainées
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #dessin du carré vert à l'emplacement du visage
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    #affichage de texte
    cv2.putText(frame,'Press ESC to stop', (0,470), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,2)

    #affichage de la frame
    cv2.imshow('Face Detector Sadeuh', frame)

    
    #attente d'une touche pour quitter le programme
    key = cv2.waitKey(1)

    #attente de echap ou q (quit)
    if key==27 or key==113:
        break

webcam.release()





print("Code completed")
