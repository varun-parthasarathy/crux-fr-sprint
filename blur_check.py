import cv2
import face_recognition as FR


def blur_check(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(image, cv2.CV_64F).var()
    return blur

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    blur = blur_check(frame)
    cv2.putText(frame, "{}: {:.2f}".format('Blur level', blur), (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()