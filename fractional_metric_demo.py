import cv2
import face_recognition as FR
from DistanceMetrics import Similarity
import numpy as np
import tensorflow as tf
from facenet import facenet

def prewhiten(image):
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
    processed = np.multiply(np.subtract(image, mean), 1/std_adj)
        
    return processed

def hist_equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    l1 = clahe.apply(l)
    processed = cv2.merge((l1, a, b))
    processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)

    return processed

cap = cv2.VideoCapture(0)
dist = Similarity()


with tf.Graph().as_default():
    with tf.Session() as session:

        facenet.load_model('20180402-114759.pb')
        img_holder = tf.get_default_graph().get_tensor_by_name(
                                    'input:0')
        embeddings = tf.get_default_graph().get_tensor_by_name(
                                    'embeddings:0')
        phase_train = tf.get_default_graph().get_tensor_by_name(
                                    'phase_train:0')

        test_image = cv2.imread('test2.jpg')
        #test_image = hist_equalize(test_image)
        (y1, x2, y2, x1) = FR.face_locations(test_image, model='hog')[0]
        test_face = cv2.resize(test_image[y1:y2, x1:x2], (160, 160))
        #test_face = hist_equalize(test_face)
        feed_dict = {img_holder:[test_face], phase_train:False}
        test_enc1 = session.run(embeddings, feed_dict=feed_dict)
        cv2.imshow('Test 1', test_face)
        test_face = prewhiten(test_face)
        feed_dict = {img_holder:[test_face], phase_train:False}
        test_enc2 = session.run(embeddings, feed_dict=feed_dict)
        cv2.imshow('Test 2', test_face)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                (y1, x2, y2, x1) = FR.face_locations(image,
                                      model='hog')[0]
                frame = hist_equalize(frame)
                face = cv2.resize(frame[y1:y2, x1:x2], (160, 160))
                #face = hist_equalize(face)
                cv2.imshow('Unprocessed', face)
                processed = prewhiten(face)
                cv2.imshow('Processed', processed)
            except:
                pass
            try:
                feed_dict = {img_holder:[face], phase_train:False}
                e1 = session.run(embeddings, feed_dict=feed_dict)
                feed_dict = {img_holder:[processed], phase_train:False}
                e2 = session.run(embeddings, feed_dict=feed_dict)
                distance = dist.fractional_distance(e1[0], test_enc1, fraction=0.5)
                print('Distance, not pre-whitened : ', distance)
                distance = dist.fractional_distance(e2[0], test_enc2, fraction=0.5)
                print('Distance, pre-whitened both: ', distance)
                print('')
            except:
                pass

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

cv2.destroyAllWindows()