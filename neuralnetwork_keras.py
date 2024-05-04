# neuralnetwork_keras.py
# Train and test basic neural network using Keras

import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


# set the root directory of this code
ROOT = os.path.dirname(os.path.abspath(__file__)) 
DATAFILE = 'data.csv'
TRAINFILE = 'Training.csv'
TESTFILE = 'Testing.csv'

def main():
    # Load training data and testing data
    training_data = np.loadtxt(os.path.join(ROOT, TRAINFILE), delimiter=',', dtype=str)
    testing_data = np.loadtxt(os.path.join(ROOT, TESTFILE), delimiter=',', dtype=str)

    #split the training data and testing data and parse them to floats
    x_train = training_data[1:, :8].astype(np.float64)
    x_test = testing_data[1:, :8].astype(np.float64)

    #split the training labels and testing labels and parse them to floats
    t_train = training_data[1:, 8].astype(np.float64)
    t_test = testing_data[1:, 8].astype(np.float64)
    
    #set the parameter labels and the numcol variable for the number of parameters
    labels = training_data[0, :]
    numCol = len(x_train[0])

    # scale the data down to floats between 0 and 1
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create neural network
    model = Sequential()
    model.add(Input(shape=(numCol,)))
    model.add(Dense(units=100, activation='relu', name='hidden1'))
    model.add(Dense(units=50, activation='relu', name='hidden2'))
    model.add(Dense(units=25, activation='relu', name='hidden3'))
    model.add(Dense(units=1, activation='sigmoid', name='output')) 
    model.summary()
    input("Press <Enter> to train this network...")

    # Compile the model using binary crossentropy for loss and RMSprop for optimizer
    model.compile(
        loss='binary_crossentropy',  
        optimizer=RMSprop(learning_rate=0.0001),
        metrics=['accuracy'])

    # Add optional callbacks for early stopping if the loss doesn't improve by at least 0.01 for 10 consecutive epochs
    callback = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=10,
        verbose=1)
    
    # update learning rate if loss does not imporve in 5 epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',  
                              factor=0.1,
                              patience=5,
                              verbose=1,
                              min_lr=1e-6)

    # Train the network with a batch size of 30 for 1000 epochs
    history = model.fit(x_train, t_train,  
                        epochs=1000,
                        batch_size=30,
                        callbacks=[callback, reduce_lr],
                        verbose=1,
                        validation_split = 0.2)

    # Test the network and print the test accuracy
    metrics = model.evaluate(x_test, t_test, verbose=0)  
    print(f'Test accuracy: {metrics[1]:0.4f}')

    y_pred = model.predict(x_test) #predict outputs for the test dataset 
    y_pred = np.round(y_pred).astype(int)

    cm = confusion_matrix(t_test, y_pred)  # Compute small confusion matrix
    print(f"Confusion matrix:\n {str(cm).replace('[', '').replace(']', '')}")

if __name__ == "__main__":
    main()