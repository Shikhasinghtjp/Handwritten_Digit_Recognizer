import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import os

# Create model directory if it doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")

# Load the MNIST dataset
print("🔄 Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape data
print("✅ Preprocessing data...")
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the CNN model
print("🧠 Building CNN model...")
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 classes for digits 0–9
])

# Compile the model
print("⚙️ Compiling model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("🚀 Training model...")
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)

# Evaluate the model
print("📊 Evaluating model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model_path = "model/mnist_cnn.h5"
print(f"💾 Saving model to {model_path}")
model.save(model_path)
print("🎉 Model training and saving complete!")
