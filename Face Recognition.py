import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import uuid
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from pathlib import Path

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# for gpu in gpus: 
#     tf.config.experimental.set_memory_growth(gpu, True)


# Function to preprocess the images
def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img


# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
l1 = L1Dist()
siamese_model = tf.keras.models.load_model('C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/MiRed5.h5', 
                                   custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})


#Capture the image and add it to the input path
faceClassif = cv2.CascadeClassifier('C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/Rostros.xml')
cap = cv2.VideoCapture(0)
while True:
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceClassif.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		
	cv2.imshow('Imagen de entrada',frame)
	if cv2.waitKey(1) & 0XFF == ord('v'):
		for (x, y, w, h) in faces: 
			faces = frame[y:y + h, x:x + w]
			faces = cv2.resize(faces,(100,100), interpolation=cv2.INTER_CUBIC)
			segmentor = SelfiSegmentation()
			green = (0, 0, 0)
			imgNoBg = segmentor.removeBG(faces, green, threshold=0.9)
			cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), imgNoBg)	
	if cv2.waitKey(1) & 0XFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

#function: pass the image through the model
def verify(model, detection_threshold, verification_threshold, input1, input2 ):
    # Build results array
    results = []

    input_img = preprocess(input1)
    validation_img = preprocess(input2)
    
    # Make Predictions 
    result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
    # results.append(result)
    return result


#function: go folder by folder sending each image to the verify function to compare them 
path = 'C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/application_data/input_image'
caption = os.path.join(path,'input_image.jpg')
def verify3(input_image):
    persons = os.listdir("C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/Data2") 
    print(persons)
    results_main = []
    for person in persons:
        folder = Path("C:/Users/artur/OneDrive/Documentos/Practica1/Redes neuronales/Data2", person)
        images= os.listdir(folder)
        count = 0
        # count_images = (len(images))
        for image in images:
            current_person = Path(folder, images[count])
            results = verify(siamese_model, 0.5, 0.5, input_image, str(current_person))
            results_main.append(results)
            count = count + 1
    print(results_main)
    max = np.max(results_main)
    max1 = results_main.index(np.max(results_main))
    max2 = int(np.floor(max1/count))
    max2 = persons[max2]
    print('La imagen de entrada tiene el maximo parecido de',max,'sobre 1 con la persona', max2)

verify3(caption)
