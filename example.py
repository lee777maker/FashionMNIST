import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from time import time

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion MNIST dataset
# Adjust the DATA_DIR to your folder or remove it to download automatically
try:
    train_full = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
except:
    # If download fails, try with download=False (assuming data exists)
    train_full = datasets.FashionMNIST('./data', train=True, download=False, transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, download=False, transform=transform)

# Split training data into train and validation sets
train_size = int(0.9 * len(train_full))
val_size = len(train_full) - train_size
train_set, val_set = random_split(train_full, [train_size, val_size])

print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")

# Create data loaders (batch size will be determined by hyperopt)
def create_data_loaders(batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Define a more complex neural network class with configurable architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_rates, batch_norm=True, activation='relu'):
        super(NeuralNetwork, self).__init__()
        
        # Select activation function
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation_fn = nn.ELU()
        else:  # tanh
            self.activation_fn = nn.Tanh()
        
        # Build the network layers dynamically
        layers = []
        prev_size = input_size
        
        for i, (layer_size, dropout_rate) in enumerate(zip(layer_sizes, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_size, layer_size))
            
            # Batch normalization (optional)
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))
                
            # Activation function
            layers.append(self.activation_fn)
            
            # Dropout 
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_size = layer_size
        
        # Output layer (10 classes for Fashion MNIST)
        layers.append(nn.Linear(prev_size, 10))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# Define training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, early_stopping_patience=5):
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    patience_counter = 0
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            # Move data to device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate epoch training loss
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss_epoch = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Move data to device
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_epoch += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        val_loss_epoch /= len(val_loader)
        val_losses.append(val_loss_epoch)
        val_accuracy = 100 * correct / total
        val_accs.append(val_accuracy)
        
        # Early stopping check
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Update learning rate based on validation accuracy
        if scheduler:
            scheduler.step(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_loss_epoch:.4f}, '
              f'Val Accuracy: {val_accuracy:.2f}% '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_accs, best_val_acc

# Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Define hyperparameter search space
space = {
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),
    'optimizer': hp.choice('optimizer', ['adam', 'adamw', 'rmsprop']),
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'hidden_layer_1': hp.choice('hidden_layer_1', [128, 256, 512, 1024]),
    'hidden_layer_2': hp.choice('hidden_layer_2', [64, 128, 256, 512]),
    'hidden_layer_3': hp.choice('hidden_layer_3', [32, 64, 128, 256]),
    'dropout_1': hp.uniform('dropout_1', 0.1, 0.5),
    'dropout_2': hp.uniform('dropout_2', 0.1, 0.5),
    'dropout_3': hp.uniform('dropout_3', 0.1, 0.5),
    'activation': hp.choice('activation', ['relu', 'leaky_relu', 'elu', 'tanh']),
    'batch_norm': hp.choice('batch_norm', [True, False]),
}

# Objective function for hyperopt
def objective(params):
    start_time = time()
    
    # Extract hyperparameters
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    optimizer_name = params['optimizer']
    num_layers = params['num_layers'] + 1  # Add 1 because we always want at least 1 layer
    hidden_layers = [
        params['hidden_layer_1'],
        params['hidden_layer_2'] if num_layers > 1 else None,
        params['hidden_layer_3'] if num_layers > 2 else None
    ]
    hidden_layers = [layer for layer in hidden_layers if layer is not None]
    
    dropout_rates = [
        params['dropout_1'],
        params['dropout_2'] if num_layers > 1 else None,
        params['dropout_3'] if num_layers > 2 else None
    ]
    dropout_rates = [rate for rate in dropout_rates if rate is not None]
    
    activation = params['activation']
    batch_norm = params['batch_norm']
    
    # Print current hyperparameters being tested
    print("\nTesting hyperparameters:")
    print(f"Layers: {num_layers}, Sizes: {hidden_layers}, Dropout: {dropout_rates}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate:.2e}, Weight decay: {weight_decay:.2e}")
    print(f"Optimizer: {optimizer_name}, Activation: {activation}, Batch Norm: {batch_norm}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size)
    
    # Initialize model
    input_size = 28 * 28  # Flattened Fashion MNIST image
    model = NeuralNetwork(
        input_size=input_size,
        layer_sizes=hidden_layers,
        dropout_rates=dropout_rates,
        batch_norm=batch_norm,
        activation=activation
    )
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Configure optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:  # rmsprop
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5, verbose=True
    )
    
    # Train model
    trained_model, _, _, _, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=30,
        early_stopping_patience=5
    )
    
    # Evaluate on test set
    test_accuracy = evaluate_model(trained_model, test_loader)
    
    # Calculate execution time
    execution_time = time() - start_time
    
    print(f"\nResults - Validation Accuracy: {best_val_acc:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Execution time: {execution_time:.2f} seconds\n")
    print("=" * 80)
    
    # We return negative accuracy since hyperopt minimizes the objective
    return {
        'loss': -best_val_acc,  # Negative because we want to maximize accuracy
        'test_accuracy': test_accuracy,
        'status': STATUS_OK,
        'eval_time': execution_time,
        'params': params
    }

# Run hyperopt optimization
def run_hyperopt(max_evals=20):
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )
    
    # Get the best parameters
    best_params = space_eval(space, best)
    
    # Print the best hyperparameters
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Get best trial
    best_trial_idx = np.argmin([r['loss'] for r in trials.results])
    best_trial = trials.results[best_trial_idx]
    
    print(f"\nBest validation accuracy: {-best_trial['loss']:.2f}%")
    print(f"Best test accuracy: {best_trial['test_accuracy']:.2f}%")
    
    return best_params, best_trial

# Helper function to convert hyperopt indexes to values
def space_eval(space, params):
    from hyperopt.pyll.base import Apply
    
    def _get_vals(space_vals, params):
        if isinstance(space_vals, dict):
            return {k: _get_vals(v, params) for k, v in space_vals.items()}
        elif isinstance(space_vals, (list, tuple)):
            return [_get_vals(v, params) for v in space_vals]
        elif isinstance(space_vals, Apply):
            if space_vals.name == 'switch':
                # For hp.choice, get the selected option
                options = [_get_vals(opt, params) for opt in space_vals.pos_args[1:]]
                return options[params[space_vals.pos_args[0].obj]]
            # For other hyperparameters, just return the value
            return params[space_vals.obj]
        return space_vals
    
    return _get_vals(space, params)

# Train with the best hyperparameters
def train_with_best_params(params):
    # Extract hyperparameters
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    optimizer_name = params['optimizer']
    
    num_layers = params['num_layers'] + 1
    hidden_layers = [
        params['hidden_layer_1'],
        params['hidden_layer_2'] if num_layers > 1 else None,
        params['hidden_layer_3'] if num_layers > 2 else None
    ]
    hidden_layers = [layer for layer in hidden_layers if layer is not None]
    
    dropout_rates = [
        params['dropout_1'],
        params['dropout_2'] if num_layers > 1 else None,
        params['dropout_3'] if num_layers > 2 else None
    ]
    dropout_rates = [rate for rate in dropout_rates if rate is not None]
    
    activation = params['activation']
    batch_norm = params['batch_norm']
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size)
    
    # Initialize model
    input_size = 28 * 28
    model = NeuralNetwork(
        input_size=input_size,
        layer_sizes=hidden_layers,
        dropout_rates=dropout_rates,
        batch_norm=batch_norm,
        activation=activation
    )
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Configure optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:  # rmsprop
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.5, verbose=True
    )
    
    # Train model
    model, train_losses, val_losses, val_accs, best_val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,  # Train longer for final model
        early_stopping_patience=7
    )
    
    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader)
    print(f"\nFinal model - Validation Accuracy: {best_val_acc:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), 'best_fashion_mnist_model.pth')
    
    # Plot training/validation curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy: {test_accuracy:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model, test_accuracy

# Visualize some predictions
def visualize_predictions(model, test_loader, num_samples=10):
    # Get batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        images_flat = images[:num_samples].reshape(-1, 28*28).to(device)
        outputs = model(images_flat)
        _, predicted = torch.max(outputs, 1)
    
    # Convert images and predictions back to CPU
    images = images[:num_samples].cpu()
    labels = labels[:num_samples].cpu()
    predicted = predicted.cpu()
    
    # Class names
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Plot the images and predictions
    fig = plt.figure(figsize=(12, 8))
    for idx in range(num_samples):
        ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
        img = images[idx].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        
        # Set title color based on prediction correctness
        title_color = 'green' if predicted[idx] == labels[idx] else 'red'
        ax.set_title(f"Pred: {classes[predicted[idx]]}\nTrue: {classes[labels[idx]]}", 
                     color=title_color)
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

# Function to generate confusion matrix
def plot_confusion_matrix(model, test_loader):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Prepare data
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
               xticklabels=['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'],
               yticklabels=['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Starting hyperparameter optimization...")
    max_evals = 15  # Number of hyperparameter configurations to try
    
    # Run hyperopt to find best hyperparameters
    best_params, best_trial = run_hyperopt(max_evals=max_evals)
    
    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    final_model, final_test_accuracy = train_with_best_params(best_params)
    
    # Create visualization of predictions
    _, _, test_loader = create_data_loaders(64)
    visualize_predictions(final_model, test_loader)
    
    # Generate confusion matrix
    plot_confusion_matrix(final_model, test_loader)
    
    print(f"\nFinal test accuracy: {final_test_accuracy:.2f}%")
    if final_test_accuracy > 95:
        print("✅ Achieved target accuracy of 95%!")
    else:
        print("❌ Did not achieve target accuracy of 95%. Consider increasing max_evals or adjusting the search space.")