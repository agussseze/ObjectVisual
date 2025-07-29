import cv2
from ultralytics import YOLO

# Cargar modelo YOLOv8 preentrenado
model = YOLO("yolov8n.pt")  # "n" es el modelo más liviano (nano)

# Capturar desde webcam
cap = cv2.VideoCapture(0)  # 0 es la webcam principal

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar detección
    results = model(frame)[0]

    # Dibujar resultados
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Dibujar rectángulo y texto
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Mostrar imagen
    cv2.imshow("Detección con YOLOv8", frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()