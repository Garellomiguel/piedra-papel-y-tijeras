import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade\Hand_haar_cascade.xml')

cap = cv2.VideoCapture(0)


def change_res(width, height):
    '''Las opciones clasicas seria (1920,1080) (1280,720) (640,480)  '''
    cap.set(3, width)
    cap.set(4, height)

change_res(640,480)

while True:

    ret, frame = cap.read()
    blur = cv2.GaussianBlur(frame,(5,5),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    retval2,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    faces = face_cascade.detectMultiScale(thresh1, scaleFactor=1.3, minNeighbors=5)

    for (x,y,w,h) in faces:
        roi_frame = frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x,y), (x + w, y+h), (255,0,0), 1)


    #ret,thresh2 = cv2.threshold(gray,160,200,cv2.THRESH_BINARY)
    #th2 = cv2.adaptiveThreshold(gray.copy(),150,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,2)
    #frame_h, frame_w = frame.shape[0:2]
    #mitad_pantalla = int(frame_w/2)
    
    #oi = frame[60:mitad_pantalla-60, 60:mitad_pantalla-60]

    #cv2.rectangle(frame, (60,60), (mitad_pantalla-60,mitad_pantalla-60), (255,0,0), 1)
    #cv2.rectangle(frame,(mitad_pantalla+60,60), (frame_w-60,mitad_pantalla-60), (0,255,0),1 )


    cv2.imshow('frame',frame)
    cv2.imshow('grey',thresh1)
    cv2.imshow('blur',blur)

    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()