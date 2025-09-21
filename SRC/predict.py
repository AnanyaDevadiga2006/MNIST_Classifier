import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# === 1️⃣ Load the MNIST model ===
model_path = r"C:\Users\Admin\OneDrive\เอกสาร\PYTHON\ML_MINIST_multiclass_classifier\models\mnist_model.h5"

if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    sys.exit(1)

model = load_model(model_path)
print("✅ Model loaded successfully!")

# === 2️⃣ Prepare images folder ===
img_folder = os.path.join(os.path.dirname(model_path), "..", "img")
if not os.path.exists(img_folder):
    print(f"❌ Image folder not found: {img_folder}")
    sys.exit(1)

# === 3️⃣ Predict all PNG images in folder ===
img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(".png")]
if not img_files:
    print(f"❌ No PNG images found in {img_folder}")
    sys.exit(1)

for img_file in img_files:
    img_path = os.path.join(img_folder, img_file)

    # Load image
    try:
        img = Image.open(img_path).convert("L")  # convert to grayscale
        img = img.resize((28, 28))               # resize to 28x28
        img_array = np.array(img, dtype=np.float32) / 255.0  # normalize
        img_array = img_array.reshape(1, 28, 28, 1)         # shape for model
    except Exception as e:
        print(f"❌ Failed to process {img_file}: {e}")
        continue

    # Predict
    pred = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(pred)

    # Print result
    print(f"🖼️ Image: {img_file}  →  🔢 Predicted digit: {predicted_digit}")

    # Optional: show image
    plt.imshow(img_array.reshape(28, 28), cmap="gray")
    plt.title(f"Prediction: {predicted_digit}")
    plt.axis("off")
    plt.show()
