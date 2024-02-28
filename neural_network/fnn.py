import numpy as np

#TODO backprop func fix, shape of weigths and deltas. 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
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
        #Although seems trivial, i made too much mistakes here so i went back
        #and make it very basic just to make it work so:
        #TODO clean and improve efficiency here
        _A = []
        _Z = []
        predictions = []
        
        for i in X:
            A = {0: i}
            Z = {0: None}
            
            #for every layer we dot layer nodes to weights and add bias
            #and apply activation
            layers = len(weights)
            for layer in range(1, layers):
                neurons = len(weights[layer])
                a_values = []
                z_values = []
                for neuron in range(neurons):
                    z = (np.sum(weights[layer][neuron] * A[layer-1]) + biases[layer][neuron]).item()
                    
                    a_values.append(activation(z))
                    z_values.append(z)
                    
                A[layer] = np.array(a_values)
                Z[layer] = np.array(z_values)
            
            #we do it for the output layer too, but dont use activation for it 
            values = []
            neurons = len(weights[layers])
            for neuron in range(neurons):   
                z = (np.sum(weights[layers][neuron] * A[layers-1]) + biases[layers][neuron]).item()
                values.append(z)
                
            #values will be predictions
            predictions.append(values)
            
            values = np.array(values)
            A[layers] = values
            Z[layers] = values
             
            _A.append(A)
            _Z.append(Z)
        
        return _A, _Z, predictions
                
    def backpropagation(self, X, predictions, true_labels, A, Z, weights, biases):
        #NN here will be available for single-target regression tasks
        #(one output neuron-continous value) for simplicities sake
        #so we assume predictions will one numerical value
        predicted = [p for sublist in predictions for p in sublist]
        
        layers = len(weights)
        deltas = [None] * layers
        grads_weights = {}
        grads_biases = {}
        
        #error for output layer
        deltas[-1] = (predicted - true_labels) / X.shape[0]
        
        for l in range(layers - 2, -1, -1):
            print(deltas[-1]) 
            
            """* (self.activation(Z[l]) * (1 - self.activation(Z[l])))"""

        return None, None
     
    def update_parameters(self, gradients, learning_rate):
        pass
         
    def train(self, X, y, epochs=1000, learning_rate=0.01, loss_fn=mse_loss):
        w, b = self.initialize_parameters(self.layer_structure)
        
        for epoch in epochs:
            A, Z, predictions = self.forward_propagation(X, y, w, b, self.activation)
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
a, z, predictions = nn.forward_propagation(X_train, w, b)
zz, zz2 = nn.backpropagation(X_train, predictions, y_train, a, z, w, b)

#print(zz)
