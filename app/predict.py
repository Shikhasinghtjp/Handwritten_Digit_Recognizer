import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# Load model once
import os
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'mnist_cnn.h5')
model_path = os.path.abspath(model_path)
model = load_model(model_path)



def predict_digit(img):

    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img)
    return np.argmax(pred), np.max(pred)
