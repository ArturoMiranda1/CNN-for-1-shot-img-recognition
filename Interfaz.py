import cv2
import os
import numpy as np
from keras.layers import Layer
import tensorflow as tf
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from pathlib import Path
from tkinter import *
from PIL import Image, ImageTk
import imutils

root = Tk()


root.title("Reconocimiento facial HGM")

root.geometry("800x600")


fondo = PhotoImage(file = 'C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/Interfaz de reconocimiento Facial.png')


fondo1 = Label(root, image = fondo).place(x = 0, y = 0, relwidth = 1, relheight = 1)


root.iconbitmap("C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/HGM-imagen.ico")

root.resizable(0,0)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    img = tf.image.resize(img, (100,100))

    img = img / 255.0

    return img

class L1Dist(Layer):
    
    def __init__(self, **kwargs):
        super().__init__()
       
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
l1 = L1Dist()
siamese_model = tf.keras.models.load_model('C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/MiRed5.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

cap = None


def deteccion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)


def captura():
    global cap
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    if ret == True:
        for (x, y, w, h) in faces: 
            faces = frame[y:y + h, x:x + w]
            faces = cv2.resize(faces,(100,100), interpolation=cv2.INTER_CUBIC)
            segmentor = SelfiSegmentation()
            green = (0, 0, 0)
            imgNoBg = segmentor.removeBG(faces, green, threshold=0.9)
            path = 'C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/application_data/input_image'
            cv2.imwrite(os.path.join(path, 'input_image.jpg'), imgNoBg)


def video_stream():
    global cap
    cap = cv2.VideoCapture(0)
    instruccion()


faceClassif = cv2.CascadeClassifier('C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/Rostros.xml')
def instruccion():
    global cap
    ret,frame = cap.read()
    if ret == True:
        etiqvid.place(x = 196, y = 77)
        frame = imutils.resize(frame, width = 450)
        deteccion(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(frame)
        image2 = ImageTk.PhotoImage(image = img2)
        etiqvid.configure(image = image2)
        etiqvid.image = image2
        etiqvid.after(10, instruccion)	


def quitar():
    global cap
    etiqvid.place_forget()
    cap.release()

def procesando():
    def verify(model, detection_threshold, verification_threshold, input1, input2 ):
        # Build results array

        input_img = preprocess(input1)
        validation_img = preprocess(input2)
        
        # Make Predictions 
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        return result
    path = 'C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/application_data/input_image'
    caption = os.path.join(path,'input_image.jpg')
    def verify3(input_image):
        persons = os.listdir("C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/Data2")
        results_main = []
        for person in persons:
            folder = Path("C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/Data2", person)
            images= os.listdir(folder)
            count = 0
            for image in images:
                current_person = Path(folder, images[count])
                results = verify(siamese_model, 0.5, 0.5, input_image, str(current_person))
                results_main.append(results)
                count = count + 1
        max = np.max(results_main)
        print(results_main)
        max1 = results_main.index(np.max(results_main))
        max2 = int(np.floor(max1/count))
        max2 = persons[max2]
        if max >= 0.9:
            text = 'La imagen de entrada tiene el maximo parecido de: {max}'
            text1 = 'Con la imagen en la posicion: {max1}'
            text2 = 'Con la persona: {max2}'
        else:
            text = 'La persona es desconocida'
            text1 = 'La persona es desconocida'
            text2 = 'La persona es desconocida'
        result = Label(text = text.format(max = max)).place(x = 196, y = 200)
        result1 = Label(text = text1.format(max1 = max1)).place(x = 196, y = 250)
        result2 = Label(text = text2.format(max2 = max2)).place(x = 196, y = 300) 

    verify3(caption)

BotonRostro = Button(root, text="Activar Camara", bg="white", relief = 'flat', width= 15, height = 2, command=video_stream)
BotonRostro.place(x=148, y=450)


BotonQuitar = Button(root, text="Quitar Camara", bg="white", relief = 'flat', width= 15, height = 2, command=quitar)
BotonQuitar.place(x=345, y=450)


BotonResultados = Button(root, text="Ver Resultados", bg="white", relief = 'flat', width= 15, height = 2, command=procesando)
BotonResultados.place(x=540, y=450)

BotonCaptura = Button(root, text="Tomar Foto", bg="white", relief = 'flat', width= 15, height = 2, command=captura)
BotonCaptura.place(x=345, y=535)


etiqvid = Label(root, bg = 'black')
etiqvid.place(x = 196, y = 77)


root.mainloop()