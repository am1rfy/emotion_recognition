# import main
import tkinter as tk
from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import filedialog
from test import call_predict


def main_window():
    # Создание окна фото
    photo_window = Tk()
    photo_window.geometry("600x480+650+250")
    photo_window.resizable(False, False)
    photo_window.title('По фото')
    
    # Загрузка изображения из компа
    def openfn(): 
        filename = filedialog.askopenfilename(title='open')
        return filename
    
    # Конвертация изображения для ткинтера
    def open_img():
        x = openfn()
        output = call_predict(x)
        emotion_text.set(output)
        print(output)
        img = Image.open(x)
        img = img.resize((450, 280))
        img = ImageTk.PhotoImage(img)
        panel = Label(photo_window, image=img)
        panel.image = img
        panel.place(x=70, y=15)  # Расположение изображения

    # Создание виджетов
    emotion_text = StringVar()
    emotion_lb = Label(borderwidth=2, relief="sunken", pady=10, font="20", textvariable=emotion_text)
    open_btn = Button(photo_window, text='Загрузить изображение', command=open_img, pady="10", font="20", fg="#ffffff",
                      bg="#708090")

    # Расположение виджетов
    open_btn.pack(side=BOTTOM, fill=X)
    emotion_lb.pack(side=BOTTOM, fill=X)


    photo_window.mainloop()


if __name__ == '__main__':
    main_window()
