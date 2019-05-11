import cv2
import numpy as np    


def prewhiten(image):
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
    processed = np.multiply(np.subtract(image, mean), 1/std_adj)
        
    return processed


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow('Unprocessed', frame)
    processed = prewhiten(frame)
    cv2.imshow('Processed', processed)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
