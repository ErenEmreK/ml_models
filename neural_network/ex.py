import numpy as np
from math import e

def sigmoid(x):
    return 1 / (1 + e ** -x)
    
# Step 1: Initialization
def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims) - 1  # Number of layers
    
    for l in range(1, L + 1):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    print(parameters)
    return parameters

# Step 2: Forward Propagation
def forward_propagation(X, parameters, activation=sigmoid):
    caches = []
    A = X
    L = len(parameters) // 2  # Number of layers
    
    for l in range(1, L):
        Z = np.dot(parameters['W' + str(l)], A) + parameters['b' + str(l)]
        A = activation(Z)
        caches.append((Z, A))
    
    Z = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = activation(Z)
    caches.append((Z, AL))
    
    return AL, caches




from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10, n_features=5, n_classes=2, 
    n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


w = initialize_parameters([5, 2, 1])
a, z = forward_propagation(X_train[1], w)

#print(z)
