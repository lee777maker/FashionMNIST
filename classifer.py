import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, io
from torch.utils.data import DataLoader
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random

#Step 1
## Image Processing

DATA_DIR = "/Users/lethaboneo/Desktop/Computer Science/CSC3022F/Assignments/Machine Learning/ML Assignment 1/"
train = datasets.FashionMNIST(DATA_DIR, train=True, download=False)
test = datasets.FashionMNIST(DATA_DIR, train=False, download=False)

## Print the datasets
print("Train dataset:", train)
print("Test dataset:", test)

X_train = train.data.float()
y_train = train.targets
X_test = test.data.float()
y_test = test.targets

item = 0
item1 =8

subsetIndicesTrain = np.where((y_train == item )|(y_train==item1))[0]
subsetIndicesTest = np.where((y_test == item )|(y_test==item1))[0]

X_train = X_train[subsetIndicesTrain]
y_train = y_train[subsetIndicesTrain]
X_test = X_test[subsetIndicesTest]
y_test = y_test[subsetIndicesTest]

def display_image(x,y):
    plt.imshow(x,cmap='binary')
    plt.title("Label: %d" % y)
    plt.show()

for i in range(3):
    display_image(X_train[i],y_train[i])


test_size = X_test.shape[0]
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

X_valid = X_train[indices]
y_valid = y_train[indices]

X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

X_train = X_train.reshape(-1, 28*28)
X_valid = X_valid.reshape(-1,28*28)
X_test = X_test.reshape(-1,28*28)

n_feat = X_train.shape[1]

## Activation function
def sigmod(r):
    return 1/(1 + torch.exp(-r))

class NeuralNetwork(nn.Module):
    def __init__(self, input_nodes, hidden_layers, output_node):
        super(NeuralNetwork,self).__init__()
        self.sig= nn.Linear(input_nodes,hidden_layers)

    def forward(self,x):
        out = self.sig(x)
        out = torch.sigmoid(x)
        return out
