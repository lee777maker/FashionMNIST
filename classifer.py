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
## Load  FashionMNIST dataset

DATA_DIR = "/Users/lethaboneo/Desktop/Computer Science/CSC3022F/Assignments/Machine Learning/ML Assignment 1/"
train = datasets.FashionMNIST(DATA_DIR, train=True, download=False)
test = datasets.FashionMNIST(DATA_DIR, train=False, download=False)

## Print the datasets
print("Train dataset:", train)
print("Test dataset:", test)

# Print the shape of the data and targets
print(train.data.shape)
print(train.targets.shape)

print(test.data.shape)
print(test.targets.shape)

# Create variables for MNIST data
X_train = train.data.float()
y_train = train.targets
X_test = test.data.float()
y_test = test.targets

def display_image(image, label):
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

    for i in range(5):
        display_image(X_train[i], y_train[i])

""" Split training data into training and validation (let validation be the size of test) """

# Sample random indices for validation
test_size = X_test.shape[0]
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

# Create validation set
X_valid = X_train[indices]
y_valid = y_train[indices]

# Remove validation set from training set
X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

# We need to reshape the data from matrices to vectors
X_train = X_train.reshape(-1, 28*28)
X_valid = X_valid.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)

n_feat = X_train.shape[1]  # Store number of features for later use (784)

# Let's start by defining our classification function p(y|x)
def sigmoid(r):
    return 1 / (1 + torch.exp(-r))

class FashionMINST(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

# Step 2
# Create data loaders
batch_size = 64
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

## Create a model
num_epochs = 5
learning_rate = 0.001
input_size = 28 * 28

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FashionMINST().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    

