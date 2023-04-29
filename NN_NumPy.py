import os
import numpy as np
import pickle as pk
import warnings
################################
directory = 'C:/DiceStuff/DiceDataSorted'
################################

warnings.filterwarnings('ignore')

print('\nLoading dataframe')
os.chdir(directory)
with open('dataframe.pickle', 'rb') as f:
    data = pk.load(f)

## CNN setup

global m , n
m , n = data.shape
np.random.shuffle(data)
data_dev = data[0:int(m*0.2)].T
y_dev = data_dev[0]
x_dev = data_dev[1:n]
x_dev = x_dev/255.

data_train = data[int(m*0.2):m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train/255.

print('\nDataframe shape: ' + str(data.shape) + '\n')

def init_params():
    W1 = np.random.rand(784,16384)
    b1 = np.random.rand(784,1)
    W2 = np.random.rand(20,784)
    b2 = np.random.rand(20,1)
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    A = e_Z / e_Z.sum(axis=0)    
    return A

def forward_prop(W1, b1, W2, b2, X):
    # print('X: ' + str(X))
    Z1 = (np.dot(W1,X) + b1)/10000.
    # print('size of Z1: ' + str(Z1.shape))
    # print('Z1: ' + str(Z1))
    A1 = ReLU(Z1)
    #A1 = np.tanh(Z1)
    # print('size of A1: ' + str(A1.shape))
    # print('A1: ' + str(A1))
    Z2 = np.dot(W2,A1) + b2
    # print('size of Z2: ' + str(Z2.shape))
    # print('Z2: ' + str(Z2))
    A2 = softmax(Z2)
    # print('size of A2: ' + str(A2.shape))
    # print('A2: ' + str(A2))
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    Y = (np.rint(Y)).astype(int)
    one_hot_Y = np.zeros((Y.size, Y.max()))
    one_hot_Y[np.arange(Y.size), Y - 1] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * np.dot(dZ2,A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = np.dot(W2.T,dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * np.dot(dZ1,X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    try:
        with open('weights&biases.pickle', 'rb') as f:
            W1, b1, W2, b2 = pk.load(f)
    except:
        W1, b1, W2, b2 = init_params()
    # W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        # print('b1 init: ' + str(b1))
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 5 == 0: ## Every 5th iteration
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print('Predictions: ' +str(predictions))
            print('Answers: ' +str(Y))
            accuracy = get_accuracy(predictions, Y)
            print('Accuracy: ' + str(accuracy))
            print('Max confidence: ' + str(np.max(A2)))
            Wb = [W1[0,0], b1[0,0], W2[0,0], b2[0,0]]
            print('W1 b1 W2 b2: ' + str(Wb) + '\n')
            # Save weights and biases
            with open('weights&biases.pickle', 'wb') as f:
                w_b = [W1, b1, W2, b2]
                pk.dump(w_b, f, pk.HIGHEST_PROTOCOL)
    return W1, b1, W2, b2

## Train model
W1, b1, W2, b2 = gradient_descent(x_train, y_train, 0.500, 500)


## Test model

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)


test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

# Test on dev data
dev_predictions = make_predictions(x_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, y_dev)