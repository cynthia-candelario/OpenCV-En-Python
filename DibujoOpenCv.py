import cv2
import numpy as np
import mediapipe as mp
import math

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
last_point = None
no_hand_counter = 0

def calcular_distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = False

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = hand.landmark

        # Coordenadas importantes
        x_tip = int(landmarks[8].x * w)
        y_tip = int(landmarks[8].y * h)
        x_pip = int(landmarks[6].x * w)
        y_pip = int(landmarks[6].y * h)

        # Punto 0 (muñeca) y 9 (centro de la palma)
        x0 = int(landmarks[0].x * w)
        y0 = int(landmarks[0].y * h)
        x9 = int(landmarks[9].x * w)
        y9 = int(landmarks[9].y * h)

        distancia_mano = calcular_distancia((x0, y0), (x9, y9))

        # Umbral: solo dibuja si la mano está cerca (distancia grande)
        umbral_cercania = 80

        dedo_extendido = y_tip < y_pip

        if dedo_extendido and distancia_mano > umbral_cercania:
            hand_detected = True
            if last_point is not None:
                cv2.line(canvas, last_point, (x_tip, y_tip), (255, 255, 255), 4)
            last_point = (x_tip, y_tip)
        else:
            last_point = None

        # Visualización
        cv2.circle(frame, (x_tip, y_tip), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Dist: {int(distancia_mano)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        if distancia_mano <= umbral_cercania:
            cv2.putText(frame, "Muy lejos - pausa", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if not hand_detected:
        no_hand_counter += 1
        if no_hand_counter > 2:
            last_point = None
    else:
        no_hand_counter = 0

    # Mostrar canvas combinado
    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)
    cv2.putText(frame, "'q' salir | 'c' limpiar", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Escribir con el dedo", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
