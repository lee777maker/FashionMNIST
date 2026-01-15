# FashionMNIST
![Dress sample](Final/fashion-jpegs/dress.jpg)
A Neural Network that predicts user fashion garments based on a trained neural network trained on MNIST-like Fashion images
# Data Processing
- Reading files from directory
- Converting the images into vectors
- Displaying the images
- Subsetting the data
- Assigning the kind of data to labels
# Neural Network Implementation
Final model is a 1 hidden layer Neural Network, Used Hyperopt to fine-tune hyperparameters.
Final model achieved 90.72% Accuracy.

Graph analysis:
![Test Confusion Matrix](Final/graphs/Test%20confusion%20Matric.png)
## How to run
1. Upload dataset in same directory as system.
2. Run python3/python classifier.py


## The classifier.py
This is the whole program that takes trains data on Fashion MINST data and predicts using Jpegs.
The System saves the model to the users computer but deletes it after the run is over
