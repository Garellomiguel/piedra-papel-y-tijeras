#librerias
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


clasificador = Sequential()

#creacion de la CNN
clasificador.add(Convolution2D(32, 3,3, input_shape =(128,128,1), activation='relu'))
clasificador.add(MaxPooling2D(pool_size=(2,2)))
clasificador.add(Convolution2D(32, 3,3, activation='relu'))
clasificador.add(MaxPooling2D(pool_size=(2,2)))

clasificador.add(Flatten())


clasificador.add(Dense(units= 128, activation='relu'))
clasificador.add(Dropout(rate = 0.1))
clasificador.add(Dense(units= 64, activation='relu'))
clasificador.add(Dropout(rate = 0.1))
clasificador.add(Dense(units= 3, activation='softmax'))

#compilacion de la CNN
clasificador.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Agregamos early stoping para evitar el overfiting
#early_stoping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


#creo los generadores para modificar cada imagen, les agrega zoom, las rota, las escala
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

#con flow for directory obtengo todas las imagenes de mi carpeta de entrenamiento ya con las etiquetas correspondientes
train_generator = train_datagen.flow_from_directory(
        '../fotos-training/training',
        target_size=(128, 128),
        color_mode = "grayscale",
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../fotos-training/test',
        target_size=(128,128),
        color_mode = "grayscale",
        batch_size=32,
            class_mode='categorical')

#ajusto el modelo con las imagenes
history = clasificador.fit(
        train_generator,
        steps_per_epoch=60,
        verbose = 1,
        epochs=85,
        validation_data=validation_generator,
        validation_steps=10)

#salvo el modelo asi no lo tengo q volver a entrenar cada vez
clasificador.save('clasificador.h5')
print('modelo salvado')

#clasificador.summary()

#grafico tanto la presicion como la perdida para ver si tengo overfiting o underfiting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


