import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, io
from torch.utils.data import DataLoader, random_split
import os
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

# Step 1: Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Step 1: Image Processing
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

# The Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers)
        self.bn1 = nn.BatchNorm1d(hidden_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layers, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Constants
input_size = 28 * 28
num_classes = 10
MODEL_PATH = 'best_model.pth'
HYPERPARAM_PATH = 'best_hyperparams.pth'

def create_model_and_train(params):
    """Function for Hyperopt to optimize"""
    # Extract hyperparameters
    batch_size = int(params['batch_size'])
    hidden_layers = int(params['hidden_layers'])
    dropout_rate = params['dropout_rate']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    
    # Prepare data loaders with current batch size
    train_loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)
    
    # Initialize model with current parameters
    model = NeuralNetwork(input_size, hidden_layers, num_classes, dropout_rate).to(device)
    
    # Setup optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    train_epochs = 10  # Use fewer epochs for hyperparameter search
    
    for epoch in range(train_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.reshape(-1, input_size).to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        scheduler.step(val_accuracy)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
    
    # Return negative accuracy since we're minimizing
    return {'loss': -best_val_acc, 'status': STATUS_OK, 'val_accuracy': best_val_acc}

def optimize_hyperparameters():
    """Run hyperparameter optimization"""
    print("Starting hyperparameter optimization...")
    
    # Define the search space
    space = {
        'batch_size': hp.quniform('batch_size', 32, 256, 32),  # Batch sizes from 32 to 256 in steps of 32
        'hidden_layers': hp.quniform('hidden_layers', 64, 512, 64),  # Hidden layer sizes from 64 to 512
        'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.7),  # Dropout rates from 0.2 to 0.7
        'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),  # Learning rates from 0.0001 to 0.01
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-4))  # Weight decay from 1e-6 to 1e-4
    }
    
    # Run the optimization
    trials = Trials()
    best = fmin(
        fn=create_model_and_train,
        space=space,
        algo=tpe.suggest,
        max_evals=15,  # Number of parameter combinations to try
        trials=trials
    )
    
    # Convert some parameters to integers
    best['batch_size'] = int(best['batch_size'])
    best['hidden_layers'] = int(best['hidden_layers'])
    
    print("\nBest hyperparameters found:")
    print(f"Batch Size: {best['batch_size']}")
    print(f"Hidden Layers: {best['hidden_layers']}")
    print(f"Dropout Rate: {best['dropout_rate']:.4f}")
    print(f"Learning Rate: {best['learning_rate']:.6f}")
    print(f"Weight Decay: {best['weight_decay']:.6f}")
    
    # Save the best hyperparameters
    torch.save(best, HYPERPARAM_PATH)
    return best

def trainModel(params=None):
    """Train the model with either provided parameters or load best parameters"""
    if params is None:
        if os.path.exists(HYPERPARAM_PATH):
            params = torch.load(HYPERPARAM_PATH)
            print("Loading best hyperparameters from previous optimization:")
            print(f"Batch Size: {params['batch_size']}")
            print(f"Hidden Layers: {params['hidden_layers']}")
            print(f"Dropout Rate: {params['dropout_rate']:.4f}")
            print(f"Learning Rate: {params['learning_rate']:.6f}")
            print(f"Weight Decay: {params['weight_decay']:.6f}")
        else:
            print("No optimized hyperparameters found. Using defaults.")
            params = {
                'batch_size': 128,
                'hidden_layers': 256,
                'dropout_rate': 0.5,
                'learning_rate': 0.001,
                'weight_decay': 1e-5
            }
    
    # Setup with the determined parameters
    batch_size = params['batch_size']
    hidden_layers = params['hidden_layers']
    dropout_rate = params['dropout_rate']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    
    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=valData, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = NeuralNetwork(input_size, hidden_layers, num_classes, dropout_rate).to(device)
    
    # Setup optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)
    
    # Training tracking variables
    train_loss = []
    val_loss = []
    val_acc = []
    best_val_acc = 0
    num_epochs = 30  # Full training epochs
    
    # Start training
    total_start_time = time.time()
    print(f"\nStarting full training with optimized parameters for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.reshape(-1, input_size).to(device)
                labels = labels.to(device)
                
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
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    total_time = time.time() - total_start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print("Done!")
    
    return model

def checkImage(image_path):
    """Check an image with the trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
    
    # Load best hyperparameters if available
    if os.path.exists(HYPERPARAM_PATH):
        params = torch.load(HYPERPARAM_PATH)
        hidden_layers = params['hidden_layers']
        dropout_rate = params['dropout_rate']
    else:
        hidden_layers = 256
        dropout_rate = 0.5
    
    # Create model with same architecture
    model = NeuralNetwork(input_size, hidden_layers, num_classes, dropout_rate).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Process image
    img = io.read_image(image_path, mode=io.ImageReadMode.GRAY)
    img = img.squeeze()  
    img = img.float() / 255.0 
    img = (img - 0.5) / 0.5  
    img = img.reshape(1, -1).to(device)
    
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return class_names[predicted.item()]

def main():
    print("Fashion MNIST Classifier with Hyperopt")
    print("-------------------------------------")
    
    # Check if model and hyperparameters exist
    model_exists = os.path.exists(MODEL_PATH)
    hyperparams_exist = os.path.exists(HYPERPARAM_PATH)
    
    if not model_exists:
        print("\n=== Model Training ===")
        
        # Ask user if they want to optimize hyperparameters
        if not hyperparams_exist:
            choice = input("Would you like to optimize hyperparameters? (y/n): ").strip().lower()
            if choice == 'y':
                best_params = optimize_hyperparameters()
                trainModel(best_params)
            else:
                print("Using default hyperparameters.")
                trainModel()
        else:
            # Use previously optimized parameters
            trainModel()
    else:
        print("Model already exists. Skipping training.")
        print("To retrain, delete the 'best_model.pth' file.")
        print("To optimize hyperparameters again, delete the 'best_hyperparams.pth' file.")
    
    # Interactive prediction loop
    print("\n=== Prediction Mode ===")
    print("Enter 'exit' to quit.")
    while True:
        try:
            image_path = input("Please enter a filepath: ").strip()
            if image_path.lower() == 'exit':
                print("Exiting...")
                break
            
            prediction = checkImage(image_path)
            print(f"Classifier: {prediction}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
    
    print("Goodbye!")

if __name__ == '__main__':
    main()