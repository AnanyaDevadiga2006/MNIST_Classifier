import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model

# -----------------------------
# 1️⃣ Absolute paths
# -----------------------------
BASE_DIR = r"C:\Users\Admin\OneDrive\เอกสาร\PYTHON\ML_MINIST_multiclass_classifier"
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_model.h5")
IMG_DIR = os.path.join(BASE_DIR, "img")

# Make sure img folder exists
os.makedirs(IMG_DIR, exist_ok=True)

# -----------------------------
# 2️⃣ Load your trained model
# -----------------------------
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# -----------------------------
# 3️⃣ Load MNIST test data
# -----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# -----------------------------
# 4️⃣ Save one example image for each digit (0–9)
# -----------------------------
saved_digits = set()
for img, label in zip(x_test, y_test):
    if label not in saved_digits:
        filename = os.path.join(IMG_DIR, f"{label}.png")
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        print(f"✅ Saved {filename}")
        saved_digits.add(label)
    if len(saved_digits) == 10:
        break

# -----------------------------
# 5️⃣ Predict digits using your model
# -----------------------------
for digit_label in range(10):
    img_path = os.path.join(IMG_DIR, f"{digit_label}.png")
    # Load image
    img = plt.imread(img_path)
    
    # MNIST images are 28x28, grayscale
    if img.ndim == 3:  # if saved as RGB
        img = img[:, :, 0]
    img = img.reshape(1, 28, 28, 1).astype("float32") / 255.0

    # Predict
    pred = model.predict(img)
    predicted_digit = np.argmax(pred)
    print(f"Image {digit_label}.png => Model predicts: {predicted_digit}")
