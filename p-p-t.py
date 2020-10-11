import numpy as np
import cv2
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import transform
from PIL import Image
import os


def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (128, 128))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

def convert(img):
   np_image = img.astype('float32')/255
   np_image = transform.resize(np_image, (128, 128,1))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


clasificador = load_model("keras/clasificador.h5")
clases = {0: 'papel', 1: 'piedra', 2: 'tijeras'}
test_datagen = ImageDataGenerator(rescale=1./255)


path = os.getcwd()

cap = cv2.VideoCapture(0)


def change_res(width, height):
    '''Las opciones clasicas seria (1920,1080) (1280,720) (640,480)  '''
    cap.set(3, width)
    cap.set(4, height)

change_res(640,480)

now = datetime.datetime.now()
delta = 5
i= 1
p = True
text_izquierda = ''
text_derecha = ''
while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = frame.shape[0:2]
    mitad_pantalla = int(frame_w/2)

    new_now = datetime.datetime.now()

    roi_izquierda = gray[60:mitad_pantalla-60, 60:mitad_pantalla-60]
    roi_derecha = gray[60:mitad_pantalla-60,mitad_pantalla+60:(mitad_pantalla*2)-60]

    if new_now > now + datetime.timedelta(seconds=delta) and new_now < now + datetime.timedelta(seconds=delta+1):
        cv2.putText(frame, '3', (mitad_pantalla-45,80), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,255),fontScale=2)

    if new_now > now + datetime.timedelta(seconds=delta+1) and new_now < now + datetime.timedelta(seconds=delta+2):
        cv2.putText(frame, '2', (mitad_pantalla-5,80), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,255),fontScale=2)

    if new_now > now + datetime.timedelta(seconds=delta+2) and new_now < now + datetime.timedelta(seconds=delta+3):
        cv2.putText(frame, '1', (mitad_pantalla+30,80), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,255),fontScale=2)

    if new_now > now + datetime.timedelta(seconds=delta+3) :
        cv2.putText(frame, 'GO', (mitad_pantalla-30,20), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,255),fontScale=2)

    if new_now > now + datetime.timedelta(seconds=delta+4) and new_now < now + datetime.timedelta(seconds=delta+7):
        #cv2.imwrite(f'{i}.png',roi_izquierda)
        #np_image = load(f'{i}.png')
        np_image_izquierda = convert(roi_izquierda)
        y_prob_izquierda = clasificador.predict(np_image_izquierda)
        y_clases_izquieda = y_prob_izquierda.argmax(axis=-1)
        print(y_prob_izquierda)
        text_izquierda = clases[y_clases_izquieda[0]]
       
        np_image_derecha = convert(roi_derecha)
        y_prob_derecha = clasificador.predict(np_image_derecha)
        y_clases_derecha = y_prob_derecha.argmax(axis=-1)
        print(y_prob_derecha)
        text_derecha = clases[y_clases_derecha[0]]
        delta = delta + 10
    cv2.putText(frame, text_izquierda , (50,400), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(255,0,0),fontScale=2)
    cv2.putText(frame, text_derecha , (50+mitad_pantalla,400), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,255),fontScale=2)
    cv2.putText(frame, 'vs' , (mitad_pantalla-30,350), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(255,255,255),fontScale=2)
   
    if (text_derecha == 'piedra' and text_izquierda == 'tijeras') or (text_derecha == 'papel' and text_izquierda == 'piedra') or (text_derecha == 'tijeras' and text_izquierda == 'papel'):
        cv2.putText(frame, 'perdedor' , (50,450), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,0),fontScale=1)

    if (text_derecha == 'piedra' and text_izquierda == 'papel') or (text_derecha == 'papel' and text_izquierda == 'tijeras') or (text_derecha == 'tijeras' and text_izquierda == 'piedra'):
        cv2.putText(frame, 'perdedor' , (50+mitad_pantalla,450), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,0),fontScale=1)

    if (text_derecha == 'piedra' and text_izquierda == 'piedra') or (text_derecha == 'tijeras' and text_izquierda == 'tijeras') or (text_derecha == 'papel' and text_izquierda == 'papel'):
        cv2.putText(frame, 'perdedor' , (50+mitad_pantalla,450), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,0),fontScale=1)
        cv2.putText(frame, 'perdedor' , (50,450), fontFace= cv2.FONT_HERSHEY_COMPLEX, color=(0,0,0),fontScale=1)



    
    cv2.rectangle(frame, (60,60), (mitad_pantalla-60,mitad_pantalla-60), (255,0,0), 1)
    cv2.rectangle(frame,(mitad_pantalla+60,60), (frame_w-60,mitad_pantalla-60), (0,0,255),1 )


    cv2.imshow('frame',frame)
    cv2.imshow('izquierda',roi_izquierda)
    cv2.imshow('derecha',roi_derecha)
    #cv2.imshow('derecha',roi_derecha)
    i=i+1
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()