import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, io
from torch.utils.data import DataLoader, random_split
import os

# Configuration
DATA_DIR = "/Users/lethaboneo/Desktop/Computer Science/CSC3022F/Assignments/Machine Learning/ML Assignment 1/"
MODEL_PATH = os.path.join(DATA_DIR, "fashion_mnist_model.pth")
CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

# Neural Network Definition
class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten input
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training Function
def train_model():
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_set = datasets.FashionMNIST(DATA_DIR, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    
    # Model setup
    model = FashionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Training model...")
    for epoch in range(10):  # Reduced epochs for demonstration
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Save trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

# Classification Function
def classify_image(model, image_path):
    try:
        # Preprocess image
        img = io.read_image(image_path, mode=io.ImageReadMode.GRAY)
        img = img.float() / 255.0  # Normalize
        img = (img - 0.5) / 0.5    # Match training normalization
        img = img.unsqueeze(0)     # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
        
        return CLASS_NAMES[predicted.item()]
    
    except Exception as e:
        return f"Error: {str(e)}"

# Main Program
def main():
    # Load or train model
    if os.path.exists(MODEL_PATH):
        model = FashionClassifier()
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded pre-trained model")
    else:
        model = train_model()
    
    # Interactive classification
    print("\nEnter image paths (type 'exit' to quit)")
    while True:
        try:
            path = input("\nImage path: ").strip().strip('"\'')
            if path.lower() == 'exit':
                break
            
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
                
            prediction = classify_image(model, path)
            print(f"Prediction: {prediction}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()