from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def Execute(File):
    Model = load_model(r"C:/Users/bjpra/Desktop/IBM_PROJECT/models/mnistCNN.h5") 
    File = "C:\\Users\\bjpra\\Desktop\\IBM_PROJECT\\data\\"+File
    Img = Image.open(File).convert("L")
    Img = Img.resize((28,28) )
    Im2Arr = np.array(Img)
    Im2Arr = Im2Arr.reshape(1,28,28,1)
    Prediction = Model.predict(Im2Arr)
    print(Prediction)
    print(np.argmax(Prediction,axis=1))
    return np.argmax(Prediction,axis=1)
            
