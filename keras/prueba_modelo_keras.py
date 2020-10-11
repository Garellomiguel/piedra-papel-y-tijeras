from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import transform


def load(image):
   np_image = np.array(image).astype('float32')/255
   np_image = transform.resize(np_image, (128, 128, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image


clasificador = load_model("clasificador.h5")

#clasificador.summary()

clases = {0: 'papel',
          1: 'piedra',
          2: 'tijeras'}

test_datagen = ImageDataGenerator(rescale=1./255)

cwd =os.path.dirname(os.getcwd())
local_path = os.path.join(cwd , 'fotos-training')
fotos_path = os.path.join(local_path, 'final')
for i in os.listdir(fotos_path):
    img_path = os.path.join(fotos_path, i)
    img = Image.open(img_path)
    np_image = load(img)

    y_prob = clasificador.predict(np_image)
    y_clases = y_prob.argmax(axis=-1)
    print(clases[y_clases[0]], i)
    print(y_prob)

