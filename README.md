# MNIST Digit Classifier

This project is a simple **multiclass classifier** that can recognize handwritten digits (0-9) using the MNIST dataset.

## How it works:
1. Loads the MNIST dataset from Keras.
2. Preprocesses the data (normalizes pixel values and flattens images).
3. Builds a neural network model with one hidden layer.
4. Trains the model for 5 epochs.
5. Evaluates the model on test data.
6. Saves the trained model as `mnist_model.h5`.
7. Can make predictions on new images.

## Usage:
1. Make sure you have Python 3.x and required libraries installed:
   ```bash
   pip install tensorflow numpy
