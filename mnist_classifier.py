# Step 1: Import Libraries
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np




# Step 2: Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 3: Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Step 4: Build the model
model = Sequential([
    Flatten(input_shape=(28,28)),  # Convert 28x28 image into 784 numbers
    Dense(128, activation='relu'), # Hidden layer
    Dense(10, activation='softmax') # Output layer (0-9)
])

# Step 5: Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Step 7: Test the model
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# Step 8: Predict a digit
sample = x_test[0].reshape(1,28,28)
prediction = np.argmax(model.predict(sample))
print("Predicted digit for first test image:", prediction)

# Step 9: Save the model
model.save("mnist_model.h5")
print("Model saved as mnist_model.h5")
