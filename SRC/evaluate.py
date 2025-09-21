import os
from tensorflow.keras.models import load_model
from utils import load_mnist_data

# Load model
model_path = os.path.join("..", "models", "mnist_model.h5")
model = load_model(model_path)
print("✅ Model loaded successfully!")

# Load data
_, _, x_test, y_test = load_mnist_data()

# Evaluate
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"📊 Test Loss: {loss:.4f}")
print(f"🎯 Test Accuracy: {acc:.4f}")
