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
from keras.preprocessing import image
from PIL import Image

emotions = ['Злой', 'Отвращение', 'В ужасе', 'Счастливый', 'Нейтральный', 'Печальный', 'Удивленный']


def prepare(img):
    IMG_SIZE = 48
    img = cv.imread(img)
    img_array = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    #faces = cv.CascadeClassifier('check_face.xml')
    #img = cv.imread(img)
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #new_array = cv.resize(gray, (IMG_SIZE, IMG_SIZE))
    #result = faces.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6)
    #return result.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def call_predict(img_path):
    model = keras.models.load_model('model/ferNet.h5')
    model.load_weights('model/fernet_bestweight.h5')
    
    faces = cv.CascadeClassifier('check_face.xml')
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    result = faces.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=1)
    try:
        new_img = img[result[0][1]:(result[0][1] + result[0][3]), result[0][0]:(result[0][0] + result[0][2])]
    except:
        return 'Не удалось распознать лицо'

    cv.imwrite('./new_image.jpg', new_img)

    preds = model.predict([prepare('new_image.jpg')])

    max_happiness = max(preds[0])

    if max_happiness == 0:
        return 'Не удалось распознать'

    for i in range(len(preds[0])):
        if preds[0][i] == max_happiness:
            return emotions[i] + ', с процентом достоверности ' + str(format(max_happiness, '.0%'))