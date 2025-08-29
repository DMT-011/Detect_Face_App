import cv2, time
import numpy as np
from mtcnn import MTCNN

def detect_faces_haar(image, face_cascade):
    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    result = image.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(result, (x,y), (x+w,y+h), (255,0,0), 2)
    return result, len(faces), time.time() - start

def detect_faces_cnn(image, cnn_model, threshold=0.7):
    start = time.time()
    detector = MTCNN()
    results = detector.detect_faces(image)
    result_img = image.copy()
    for res in results:
        x, y, w, h = res['box']
        cv2.rectangle(result_img, (x,y), (x+w,y+h), (0,255,0), 2)
    return result_img, len(results), time.time() - start

