import cv2
import dlib
import face_recognition as FR
import numpy as np
from imutils import face_utils

predictor = dlib.shape_predictor('sp_68_point.dat')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FR.face_locations(gray,
                              model='hog')
    for (t, r, b, l) in faces:
        rect = dlib.rectangle(l, t, r, b)
        points = predictor(gray, rect)
        points = face_utils.shape_to_np(points)
        for (x, y) in points:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Image', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
