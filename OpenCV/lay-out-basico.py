import numpy as np
import cv2

"""primera preba con openCV. Viendo como las funciones bascias"""

cap = cv2.VideoCapture(0)


def change_res(width, height):
    '''Las opciones clasicas seria (1920,1080) (1280,720) (640,480)  '''
    cap.set(3, width)
    cap.set(4, height)

change_res(640,480)

while True:

    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,thresh2 = cv2.threshold(gray,160,200,cv2.THRESH_BINARY)
    frame_h, frame_w = frame.shape[0:2]
    mitad_pantalla = int(frame_w/2)
    
    roi = frame[60:mitad_pantalla-60, 60:mitad_pantalla-60]

    cv2.rectangle(frame, (60,60), (mitad_pantalla-60,mitad_pantalla-60), (255,0,0), 1)
    cv2.rectangle(frame,(mitad_pantalla+60,60), (frame_w-60,mitad_pantalla-60), (0,255,0),1 )


    cv2.imshow('frame',frame)
    cv2.imshow('thresh',roi)

    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()