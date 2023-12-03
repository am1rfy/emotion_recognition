import cv2 as cv
import keras
import tensorflow as tf
from main import cap_img
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop, SGD
from keras import regularizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
import datetime
import matplotlib.pyplot as plt
from keras.utils import plot_model

emotions = ['Злой', 'Отвращение', 'В ужасе', 'Счастливый', 'Нейтральный', 'Печальный', 'Удивленный']

def get_model(input_size, classes=7):
    model = tf.keras.models.Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_size))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    return model

def call_predict(img_path):
    # Загрузка обученной модели
    ROW, COL = 48, 48
    CLASSES = 7
    model = get_model((ROW, COL, 1), CLASSES)
    model.summary()
    model.load_weights('model/ferNet.h5')
    
    # Получение изображения квадрата лица из полного изображения 
    image = cap_img(img_path)
    
    # Преобразование изображения в формат RGB
    img_array = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Нормализация изображения
    IMG_SIZE = 48
    resized_image = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
    prepared_img = resized_image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # Предсказывание класса с помощью модели
    predictes = model.predict([prepared_img])[0]

    # Получение индекса класса с наибольшей вероятностью
    max_prediction = max(predictes)

    # Определение названия класса по индексу и вывод результата на экран
    for i in range(len(predictes)):
        if predictes[i] == max_prediction:
            predicted_emotion = emotions[i]
            confidence = max_prediction
            return predicted_emotion + ', с процентом достоверности ' + str(format(confidence, '.0%'))

# print(call_predict('imgs/1.jpg'))
