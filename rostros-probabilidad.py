import cv2
import dlib

# Cargar el detector de rostros de DLib
detector = dlib.get_frontal_face_detector()

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el frame.")
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = detector(gray)

    # Iterar sobre las caras detectadas
    for face in faces:
        # Obtener las coordenadas del rectángulo del rostro
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Dibujar el rectángulo del rostro en la imagen
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Mostrar la probabilidad de que sea un rostro
        probability = f"Probabilidad: {round(face.confidence, 2)}"
        cv2.putText(frame, probability, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Mostrar el frame con los rostros detectados
    cv2.imshow('Reconocimiento de Rostros', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
