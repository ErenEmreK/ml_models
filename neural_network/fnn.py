import numpy as np
from math import e

def sigmoid(x):
        return 1 / (1 + e ** -x)
    
class NeuralNetwork:
    def __init__(self, layer_structure):
        self.layer_structure = layer_structure
    
    def initialize_parameters(self, layer_structure):
        #we get layer dims as a list f.e [2, 3, 1] means we have 2 3 1 nodes in
        #each layer respectively, 2 will be feature_count since its first layer
        
        weights = {}
        biases = {}
                
        for l in range(1, len(layer_structure)):
            weights[l] = np.random.randn(layer_structure[l], layer_structure[l-1]) * 0.01
            biases[l] = np.zeros((layer_structure[l], 1))
        
        return weights, biases

    def forward_propagation(self, X, weights, biases, activation=sigmoid):
        
        a_vals = {} 
        z_vals = {}
        
        a_vals[0] = X
        z_vals[0] = X
        
        length = len(weights)
        #for every layer we dot layer nodes to weights and add bias
        #and apply activation
        for l in range(1, length):
            z = np.dot(weights[l], a_vals[l-1]) + biases[l]    
            
            a_vals[l] = activation(z)
            z_vals[l] = z
            
        z_vals[length] = np.dot(weights[length], a_vals[length-1]) + biases[length]
        a_vals[length] = z_vals[length]
        
        return a_vals, z_vals
        
    def train(self, X, y):
        weights, biases = self.initialize_parameters(self.layer_structure)
        
    
    
    
    
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10, n_features=5, n_classes=2, 
    n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nn = NeuralNetwork()
w, b= nn.initialize_parameters([5, 2, 1])
a, z = nn.forward_propagation(X_train[1], w, b)

#print(z)
