import numpy as np
import cv2
import time
import datetime
import os

"""Programa para la toma de imagenes de forma de poder usarlas en el modelo"""


dir_name = 'fotos-training'
abs_path = os.path.join(os.getcwd(),dir_name)
if not os.path.exists(abs_path):
    os.mkdir(os.path.join(os.getcwd(),dir_name))

cap = cv2.VideoCapture(0)

def change_res(width, height):
    '''Las opciones clasicas seria (1920,1080) (1280,720) (640,480)  '''
    cap.set(3, width)
    cap.set(4, height)

change_res(640,480)
i= 800

now = datetime.datetime.now()

while True :

    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    th2 = cv2.adaptiveThreshold(gray.copy(),150,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,2)
    frame_h, frame_w = frame.shape[0:2]
    mitad_pantalla = int(frame_w/2)

    #establecer region de interes para tomar las fotos
    roi_derecha = frame[60:0+mitad_pantalla-60,mitad_pantalla-60:mitad_pantalla+mitad_pantalla-60]
    roi_izquierda = frame[60:mitad_pantalla, 60:mitad_pantalla]
    img = cv2.resize(roi_izquierda, (128,128))

    delta = 20  #segundos
    new_now = datetime.datetime.now()
    papel_time = new_now < (now + datetime.timedelta(seconds=delta))
    piedra_time = new_now > (now + datetime.timedelta(seconds=delta)) and new_now < (now + datetime.timedelta(seconds=delta*2))
    tijeras_time = new_now > (now + datetime.timedelta(seconds=delta*2)) and new_now < (now + datetime.timedelta(seconds=delta*3))

    path = 'training'  #especifica si son para el training o el test

    #de esta forma tengo delta tiempo para cada tipo de categoria asi salvo siempre la misma cantidad de imagenes, tambien cambio el texto en pantalla asi se cuando cambiar de cat
    if papel_time:
        ppt = 'papel'
        if new_now > (now + datetime.timedelta(seconds=2)):
            cv2.imwrite(f'../fotos-training/{path}/papel/{i}.png',img)
    elif piedra_time:
        ppt= 'piedra'
        if (new_now < (now + datetime.timedelta(seconds=(delta*2)-1)) and new_now > (now + datetime.timedelta(seconds=delta+2))):
            cv2.imwrite(f'../fotos-training/{path}/piedra/{i}.png',img)
    elif tijeras_time:
        ppt = 'tijeras'
        if (new_now < (now + datetime.timedelta(seconds=delta*3)) and new_now > (now + datetime.timedelta(seconds=(delta*2)+2))):
            cv2.imwrite(f'../fotos-training/{path}/tijeras/{i}.png',img)
    else:
        break

    #el sleep aca sirve para no guardar todos los frames sino que espere una decima de segundo entre cuadro y cuadro
    time.sleep(0.1)
    cv2.putText(frame, f'{ppt}', (50,50), fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale= 1, color= (0,0,255))
    i = i+1
    
    
    cv2.imshow('derecha',roi_izquierda)
    cv2.imshow('frame',frame)
    
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()