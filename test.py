import cv2 as cv
import keras
import tensorflow as tf
from main import cap_img

emotions = ['Злой', 'Отвращение', 'В ужасе', 'Счастливый', 'Нейтральный', 'Печальный', 'Удивленный']

def prepare(img):
    IMG_SIZE = 48
    img_array = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def call_predict(img_path):
    img = cap_img(img_path)

    x = tf.keras.Input(shape=(48, 48, 1))
    y = tf.keras.layers.Dense(16, activation='softmax')(x)
    model = tf.keras.Model(x, y)
    model.summary()

    model = keras.models.load_model('model/ferNet.h5')

    preds = model.predict([prepare(img)])

    emotion = max(preds[0])
    confidence = emotions[preds[0].argmax()]
    
    return f'Предсказанное эмоциональное состояние: {emotion}, Процент уверенности: {confidence * 100:.2f}%'

print(call_predict('imgs/1.jpg'))
