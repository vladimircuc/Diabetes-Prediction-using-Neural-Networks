# Import necessary libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from captum.attr import IntegratedGradients
from sklearn.metrics import ConfusionMatrixDisplay
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# Define file paths and root directory
ROOT = os.path.dirname(os.path.abspath(__file__))  # Get the directory where this script is located
TRAINFILE = 'Training.csv'  # Name of the training data file
TESTFILE = 'Testing.csv'  # Name of the testing data file

def show_accuracy_evolution(accuracy_count):
    #create a plot for an accuracy evolution during training phase
    epochs = np.arange(accuracy_count.size)
    plt.xlabel("Epochs")  
    plt.ylabel("Accuracy")  
    plt.title("Evolution of Accuracy while trainig")  
    plt.plot(epochs, accuracy_count)
    plt.show()

def show_loss_evolution(loss_count):
    #creating a plot for a loss evolution visualization during the training phase
    epochs = np.arange(loss_count.size)
    plt.xlabel("Epochs")  
    plt.ylabel("Loss")  
    plt.title("Evolution of Loss while trainig")  
    plt.plot(epochs, loss_count)
    plt.show()

def show_colorful_confusion_matrix(t_np, y_pred_np):
    #create a plot for a more accurate and depictive confusion matrix 
    ConfusionMatrixDisplay.from_predictions(t_np, y_pred_np, display_labels= ["healthy", "sick"])
    plt.show()

def calculate_importaces(model, x_test, labels, device):
    model.eval()
    example_input = x_test[:1]  
    example_input_tensor = torch.tensor(example_input, dtype=torch.float32).to(device)
    example_input_tensor.requires_grad = True  
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)
    # Calculate the attributions of the input features for the target prediction
    attributions, _ = ig.attribute(example_input_tensor, target=0, return_convergence_delta=True)
    attributions = attributions.detach().cpu().numpy()  # Move attributions to CPU and convert to numpy array
    # Visualize the feature importances
    visualize_importances(labels[:-1], attributions[0])


def visualize_importances(feature_names, importances, title="Feature Importances"):
    #print graph with the importance of each input
    y_pos = np.arange(len(feature_names))
    plt.bar(y_pos, importances, align='center', alpha=0.5)
    plt.xticks(y_pos, feature_names, rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.title(title)
    plt.show()

def main():
    #uncomment the 2 lines below to set a seed for this program
    #torch.manual_seed(3520)  
    #np.random.seed(3520)  

    # Load dataset from CSV files
    training_data = np.loadtxt(os.path.join(ROOT, TRAINFILE), delimiter=',', dtype=str)  # Load training data as strings initially
    testing_data = np.loadtxt(os.path.join(ROOT, TESTFILE), delimiter=',', dtype=str)  # Load testing data as strings

    # Extract labels and features from the loaded data




    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    numCol = len(x_train[0])  # Number of features

    # Scale features to be between 0 and 1
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert scaled feature arrays to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.long)
    t_test = torch.tensor(t_test, dtype=torch.long)

    # Determine the computing device (CPU, CUDA GPU, or MPS)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    # Define the neural network architecture
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()  # Flatten input tensors
            self.model = nn.Sequential(  # Define the sequential model
                nn.Linear(numCol, 100),  # Input layer to hidden layer with 100 neurons
                nn.ReLU(),  # Activation function
                nn.Linear(100, 50),  
                nn.ReLU(),  
                nn.Linear(50, 25), 
                nn.ReLU(),  
                nn.Linear(25, 10),  # Hidden layer to output layer with 1 neuron
            )

        def forward(self, x):
            x = self.flatten(x)  # Flatten the input
            y = self.model(x)  # Pass the input through the model
            return y

    # Initialize the model and move it to the appropriate device
    model = NeuralNetwork().to(device)
    print(model)

    # Prompt user to start training
    input("Press <Enter> to train this network...")
    loss_fn = nn.CrossEntropyLoss()  # Binary cross-entropy loss for binary classification
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.0001)  # Optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=1e-6) 
    # LR scheduler to reduce learning rate on plateau

    loss_count = np.array([])
    accuracy_count = np.array([])
    num_epochs = 100  # Number of epochs for training
    for epoch in range(num_epochs):
        outputs = model(x_train)  # Forward pass: compute predicted outputs by passing inputs to the model
        
        loss = loss_fn(outputs, t_train)  # Calculate loss

        loss_count = np.append(loss_count, loss.item()) #appending the epoch loss to the list of training epochs

        #computing accuracy for this epoch
        #y_pred_train = (outputs.squeeze() > 0.5).float()  # Apply threshold to get binary class labels
        #y_pred_train_np = y_pred_train.detach().numpy()  # Convert predictions to numpy array for evaluation
        #t_np_train = t_train.detach().numpy()  # Convert true labels to numpy array for evaluation

        #accuracy_train = np.mean(y_pred_train_np == t_np_train)  # Compute accuracy for this epoch
        #accuracy_count = np.append(accuracy_count, accuracy_train.item()) #appending the epoch accuracy to the list of accuracies 

        # Adjust learning rate based on loss plateau
        scheduler.step(loss)

        optimizer.zero_grad()  # Zero the gradients before running the backward pass
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)

        if (epoch + 1) % 5 == 0:  # Print loss every 5 epochs
            with torch.no_grad():
                y_pred_train = outputs.argmax(dim=1)  # Get predicted classes
                accuracy_train = (y_pred_train == t_train).float().mean()  # Calculate accuracy
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy_train.item():.4f}')
            
        # Testing the trained network
    outputs = model(x_test)  # Forward pass to get outputs
    y_pred = outputs.argmax(dim=1)  # Predicted classes
    accuracy = (y_pred == t_test).float().mean()  # Calculate accuracy
    print(f'Accuracy = {accuracy.item():0.4f}')  # Print the accuracy

    # Confusion Matrix Calculation
    cm = confusion_matrix(t_test.numpy(), y_pred.numpy())  # Use numpy arrays for sklearn compatibility
    print(f"Confusion Matrix:\n{cm}")

    #uncomment the line below for a visualization of the accuracy evolution 
    #show_accuracy_evolution(accuracy_count)

    #uncomment the line below for a graph of the loss evolution
    #show_loss_evolution(loss_count)
    
    #uncomment this line below for a graph of the confusion matrix
    #show_colorful_confusion_matrix(t_np, y_pred_np)

    #uncomment next line for a visualization of the importance of each input
    #calculate_importaces(model, x_test, labels, device)
if __name__ == "__main__":
    main()






