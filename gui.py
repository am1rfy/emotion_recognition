# import main
import tkinter as tk
from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import filedialog
from test import call_predict


def main_window():
    # Создание главного окна
    root = tk.Tk()
    root.title("OurAI")
    root.geometry("600x480+650+250")
    root.resizable(False, False)

    def photo_window(event):  # Событие для открытия окна фото
        root.destroy()
        by_photo()

    def camera_window(event):  # Событие для открытия окна камеры
        root.destroy()
        by_camera()

    canvas = Canvas(root, width=600, height=350)  # Вставка изображения
    canvas.pack()
    home_page_pic = PhotoImage(file="imgs/homepagepic.png")
    canvas.create_image(300, 175, image=home_page_pic)  # Тут размер

    # Создание кнопок
    camera_btn = tk.Button(root, text="По камере", fg="#ffffff", bg="#708090", pady="10", font="20")
    photo_btn = tk.Button(root, text="По фото", fg="#ffffff", bg="#708090", pady="10", font="20")

    # События для кнопок
    camera_btn.bind('<Button->', camera_window)
    photo_btn.bind('<Button->', photo_window)

    # Расположения кнопок
    camera_btn.pack(side=BOTTOM, fill=X)
    photo_btn.pack(side=BOTTOM, fill=X)

    tk.mainloop()


def by_photo():
    # Создание окна фото
    photo_window = Tk()
    photo_window.geometry("600x480+650+250")
    photo_window.resizable(False, False)
    photo_window.title('По фото')


    def back_to_main(event):  # Событие кнопки назад
        photo_window.destroy()
        main_window()

    def openfn():  # Загрузка изображения из компа
        filename = filedialog.askopenfilename(title='open')
        return filename

    def open_img():  # Конвертация изображения для ткинтера
        x = openfn()
        output = call_predict(x)
        emotion_text.set(output)
        print(output)
        img = Image.open(x)
        img = img.resize((450, 280))
        img = ImageTk.PhotoImage(img)
        panel = Label(photo_window, image=img)  # Сюда вставляется изображение
        panel.image = img
        panel.place(x=70, y=15)  # Расположение изображения

    # Создание виджетов
    emotion_text = StringVar()
    emotion_lb = Label(borderwidth=2, relief="sunken", pady=10, font="20", textvariable=emotion_text)
    back_btn = Button(photo_window, text="Назад", pady="10", font="20", fg="#ffffff", bg="#708090")
    open_btn = Button(photo_window, text='Загрузить изображение', command=open_img, pady="10", font="20", fg="#ffffff", bg="#708090")

    # Расположение виджетов
    back_btn.pack(side=BOTTOM, fill=X)
    open_btn.pack(side=BOTTOM, fill=X)
    emotion_lb.pack(side=BOTTOM, fill=X)

    back_btn.bind('<Button->', back_to_main)  # Событие кнопки назад

    photo_window.mainloop()


def by_camera():  # Нужно подогнать раземер камеры, не знаю где это делается
    cap = cv.VideoCapture(0)
    cap.set(3, 500)
    cap.set(4, 300)
    faces = cv.CascadeClassifier('check_face.xml')

    # Создание окна камеры
    camera_window = Tk()
    camera_window.geometry("600x480+650+250")
    # camera_window.resizable(False, False)
    camera_window.title('По камере')

    lmain = tk.Label(camera_window) # Место, куда вставляется изображение
    lmain.pack()

    def back_to_main(event): # Событие кнорки назад
        camera_window.destroy()
        main_window()

    def show_frame(): # Твой код
        _, frame = cap.read()
        img_face = frame
        frame = cv.flip(frame, 1)
        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        result = faces.detectMultiScale(cv2image, scaleFactor=1.07, minNeighbors=4)

        for (x, y, w, h) in result:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        # output = call_predict(img_face)
        # position = ((result[0][1]*2+result[0][3])//2, result[0][0])
        # cv.putText(frame, output, position, cv.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)
        cv2image = cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image).resize((600, 440)) # Конвертирует изображение для поддержки ткинтером
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)
        camera_window.mainloop()

    back_btn = Button(camera_window, text="Назад", pady="10", font="20", fg="#ffffff", bg="#708090") # Кнопка назад
    back_btn.bind('<Button->', back_to_main) # Событие для кнопки
    back_btn.pack(side=BOTTOM, fill=X) # Расположение кнорки

    show_frame()



if __name__ == '__main__':
    main_window()
