import numpy as np
from math import e

def sigmoid(x):
        return 1 / (1 + e ** -x)
 
def mse_loss(predictions, y):
    pass
   
class NeuralNetwork:
    def __init__(self, layer_structure, activation=sigmoid):
        self.layer_structure = layer_structure
        self.activation = activation
    
    def initialize_parameters(self, layer_structure):
        #we get layer dims as a list f.e [2, 3, 1] means we have 2 3 1 nodes in
        #each layer respectively, 2 will be feature_count since its first layer
        
        #even though storing them as arrays is possible dics are better fit for nns
        weights = {}
        biases = {}
                
        for l in range(1, len(layer_structure)):
            weights[l] = np.random.randn(layer_structure[l], layer_structure[l-1]) * 0.01
            biases[l] = np.zeros((layer_structure[l], 1))
        
        return weights, biases

    def forward_propagation(self, X, weights, biases, activation=sigmoid):
        
        A = {0: X}
        Z = {0: None}
        predictions = []
        
        length = len(weights)
        #for every layer we dot layer nodes to weights and add bias
        #and apply activation
        for l in range(1, length):
            z = np.dot(weights[l], A[l-1]) + biases[l]    
            
            A[l] = activation(z)
            Z[l] = z
            
        Z[length] = np.dot(weights[length], A[length-1]) + biases[length]
        A[length] = Z[length]
        
        return A, Z
    
    def backpropagation(self, predictions, true_labels):
        pass
     
    def update_parameters(self, gradients, learning_rate):
        pass
         
    def train(self, X, y, epochs=1000, learning_rate=0.01, loss_fn=mse_loss):
        w, b = self.initialize_parameters(self.layer_structure)
        
        for epoch in epochs:
            A, Z = self.forward_propagation(X, y, w, b, self.activation)
            loss = loss_fn(A, y)
    
            print(f"Epoch: {epoch}, Loss: {loss}")
            
            gradients = self.backpropagation()
            self.update_parameters(gradients, learning_rate)
         
         
         
         
         
    
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10, n_features=5, n_classes=2, 
    n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nn = NeuralNetwork([5, 3, 1])
w, b = nn.initialize_parameters(nn.layer_structure)
a, z = nn.forward_propagation(X_train[1], w, b)

print(a)
