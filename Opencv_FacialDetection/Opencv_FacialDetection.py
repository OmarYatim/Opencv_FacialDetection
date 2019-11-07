import cv2
import numpy as np

Face_Cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_Cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('téléchargé.png')
cap = cv2.VideoCapture(0)
centers = []

while True : 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = Face_Cascade.detectMultiScale(gray, 1.3, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+h]
        eyes = eye_Cascade.detectMultiScale(roi_gray)
        x1 = x + (w/2)
        y1 = y + (h/2)
        centers.append((x1,y1))
    j=0

    while(j<len(centers)) :
          cv2.circle(frame, (int(centers[j][0]),int(centers[j][1])) ,5, (0,255,0), 1)
          j += 1




    cv2.imshow('3leh?', frame)

    if(cv2.waitKey(1) ==  27):
        break

cap.release()
cv2.destroyAllWindows()