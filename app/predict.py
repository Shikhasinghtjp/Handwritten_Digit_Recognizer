import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# Load model once
model = load_model('../model/mnist_cnn.h5')


def predict_digit(img):
    # Convert to grayscale and resize
    img = img.convert('L')  # grayscale
    img = ImageOps.invert(img)  # make black digit on white
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img)
    return np.argmax(pred), np.max(pred)
