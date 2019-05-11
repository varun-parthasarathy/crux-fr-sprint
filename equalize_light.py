import cv2
import face_recognition as FR


def hist_equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l1 = clahe.apply(l)
    processed = cv2.merge((l1, a, b))
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

    return processed


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow('Unprocessed', frame)
    processed = hist_equalize(frame)
    cv2.imshow('Processed', processed)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
