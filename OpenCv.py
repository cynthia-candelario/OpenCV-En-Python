import cv2

# Abre la cámara (el 0 es el índice de la cámara por defecto)
cap = cv2.VideoCapture(0)

# Verifica si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

while True:
    # Captura frame por frame
    ret, frame = cap.read()

    # Si no se pudo capturar el frame, salimos
    if not ret:
        print("Error: No se pudo recibir el frame.")
        break

    # Muestra el frame en una ventana llamada "Camera"
    cv2.imshow('Camera', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas al finalizar
cap.release()
cv2.destroyAllWindows()