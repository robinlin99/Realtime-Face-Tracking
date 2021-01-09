import numpy as np
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox


cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, confidences = cv.detect_face(frame) 
    print(confidences)
    cv2.imshow('frame',gray)

    for face in faces:
    	(startX,startY) = face[0],face[1]
    	(endX,endY) = face[2],face[3]
    	face_crop = np.copy(frame[startY:endY, startX:endX])
    	label, confidence = cv.detect_gender(face_crop)
    	idx = np.argmax(confidence)
    	label = label[idx]
    	label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
    	print(label)
    	cv2.imshow('frame', cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()