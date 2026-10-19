from ultralytics import YOLO
import cv2

model = YOLO('weights/best.pt')

cap = cv2.VideoCapture(0)

# NOUVEAU : Forcer la webcam à utiliser une résolution HD (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# NOUVEAU : Créer une fenêtre qu'on peut redimensionner ou mettre en plein écran
cv2.namedWindow("Mon IA ASL en direct", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mon IA ASL en direct", 1280, 720)

print("Webcam activée ! Fais un signe de la main. Appuie sur 'q' pour quitter.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erreur de lecture de la webcam.")
        break

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("Mon IA ASL en direct", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()