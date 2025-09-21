# MNIST Multiclass Classifier 

This project trains a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) using the MNIST dataset.

## Features
- Train a multiclass CNN on MNIST
- Evaluate test accuracy
- Predict new handwritten images
- Save and load trained model
- Visualize training accuracy & loss

## Project Structure
MNIST_Classifier/
│── src/          # Python scripts
│── models/       # Saved model
│── img/          # Sample images for predictions & plots
│── notebooks/    # Optional notebooks
│── README.md
│── requirements.txt
│── .gitignore

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/MNIST_Classifier.git
cd MNIST_Classifier
pip install -r requirements.txt
Usage
bash
Copy code
# Train the model
python src/train.py

# Evaluate model
python src/evaluate.py

# Predict a custom image
python src/predict.py img/3.png
Note: Input images should be grayscale 28×28 pixels.

Future Improvements
Add more CNN layers

Use EMNIST (letters + digits)

Deploy as a web app

License
MIT License

yaml
Copy code

---

If you want, I can **also create the ready-to-use `img/` folder with digits 0–9** so that anyone cloning your repo can run `predict.py` immediately without errors.  

