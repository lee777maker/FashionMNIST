import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, io
from torch.utils.data import DataLoader, random_split
import os

# Step 1: Image Processing
## This is all code I referenced from the tutorial.
DATA_DIR = "."
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train = datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)
test = datasets.FashionMNIST(DATA_DIR, train=False, download=False, transform=transform)
train_size = int(0.9 * len(train))
val_size = len(train) - train_size
trainData, valData = random_split(train, [train_size, val_size])
##
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
## END OF Tutorial reference

# The Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers)
        self.bn1 = nn.BatchNorm1d(hidden_layers)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

#Hyperparameters all received 
learn_rate = 0.001
num_epochs = 30
input_size = 28 * 28
hidden_layers = 256
num_classes = 10


model = NeuralNetwork(input_size, hidden_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

#file name for the best model
MODEL_PATH = 'best_model.pth'


train_loss = []
val_loss = []
val_acc = []
best_val_acc = 0

#Training phase of model
def trainModel():
    global best_val_acc
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.reshape(-1, input_size)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        model.eval()
        val_loss_epoch = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.reshape(-1, input_size)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_epoch += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss_epoch /= len(val_loader)
        val_loss.append(val_loss_epoch)
        val_accuracy = 100 * correct / total
        val_acc.append(val_accuracy)
        
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f"Model saved with validation accuracy: {val_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Done!")

def checkImage(image_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
    
    #converts jpeg into 1d array
    img = io.read_image(image_path, mode=io.ImageReadMode.GRAY)
    img = img.squeeze()  
    img = img.float() / 255.0 
    img = (img - 0.5) / 0.5  
    img = img.reshape(1, -1) 
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return class_names[predicted.item()]

def main():
    # Check if model exists
    model_exists = os.path.exists(MODEL_PATH)
    
    if not model_exists:
        print("Pytorch Output...")
        print("...")
        trainModel()
    else:
        print("Model already exists. Skipping training.")
        print("To retrain, delete the 'best_model.pth' file.")
    
    # Interactive prediction loop
    print("Enter 'exit' to quit.")
    while True:
        try:
            image_path = input("Please enter a filepath: ").strip()
            if image_path.lower() == 'exit':
                print("Exiting...")
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                    print("Goodbye")
                break
            
            prediction = checkImage(image_path)
            print(f"Classifier: {prediction}")
            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()