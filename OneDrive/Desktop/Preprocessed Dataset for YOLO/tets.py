import cv2

print("Recherche des caméras disponibles...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Caméra trouvée sur le port : {i}")
        cap.release()