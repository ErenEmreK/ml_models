import numpy as np
from math import e

def sigmoid(x):
        return 1 / (1 + e ** -x)
    
class NeuralNetwork:
    def __init__():
        pass
    
    def initialize_parameters(layer_structure):
        #we get layer dims as a list f.e [2, 3, 1] means we have 2 3 1 nodes in
        #each layer respectively, 2 will be feature_count since its first layer
        
        weights = [[1 for _ in range(layer_structure[0])]]
        biases = [[0 for _ in range(layer_structure[0])]]
        
        for l in range(1, len(layer_structure)):
            weights.append(np.random.randn(layer_structure[l], layer_structure[l-1]) * 0.01)
            biases.append(np.zeros((layer_structure[l], 1)))
        
        return weights, biases

    def forward_propagation(X, weights, bias, activation=sigmoid):
        
        pass
    
    
s = NeuralNetwork.initialize_parameters([2, 3, 1])
print(s[0])