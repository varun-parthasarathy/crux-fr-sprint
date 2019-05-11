import cv2
import face_recognition as FR
from imutils import face_utils
import dlib

capture = cv2.VideoCapture(0)

def hist_equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    l1 = clahe.apply(l)
    processed = cv2.merge((l1, a, b))
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)

    return processed

predictor = dlib.shape_predictor('sp_68_point.dat')
aligner = face_utils.FaceAligner(predictor, desiredFaceWidth=300)

while True:
    ret, frame = capture.read()
    if not ret:
        continue
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = hist_equalize(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = FR.face_locations(image,
                              model='hog')
    for (t, r, b, l) in faces:
        rect = dlib.rectangle(l, t, r, b)
        points = predictor(image, rect)
        points = face_utils.shape_to_np(points)
        face = aligner.align(frame[:, :, ::-1], image, rect)
        '''cv2.rectangle(frame, (l, t),
                     (r, b),
                     (0, 255, 0), 2)'''
        for (x, y) in points:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        cv2.imshow('Face', face[:, :, ::-1])
    cv2.imshow('Webcam Feed', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()