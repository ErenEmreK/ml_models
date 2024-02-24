import numpy as np

class FeedforwardNeuralNetwork:
    def __init__(self, layer_dims, activation_function='sigmoid'):
        self.layer_dims = layer_dims
        self.activation_function = activation_function
        self.parameters = self.initialize_parameters()
        
    def initialize_parameters(self):
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
        return parameters
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def activation(self, Z):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(Z)
        elif self.activation_function == 'relu':
            return self.relu(Z)
    
    def forward_propagation(self, X):
        A = X
        caches = []
        for l in range(1, len(self.layer_dims)):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A) + b
            A = self.activation(Z)
            caches.append((A, W, b, Z))
        return A, caches
    
    def compute_loss(self, Y_hat, Y):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / m
        return loss
    
    def backward_propagation(self, Y_hat, Y, caches):
        gradients = {}
        m = Y.shape[1]
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
        for l in reversed(range(1, len(self.layer_dims))):
            A, W, b, Z = caches[l - 1]
            dZ = dA_prev * self.activation(Z) * (1 - self.activation(Z))
            gradients['dW' + str(l)] = np.dot(dZ, A.T) / m
            gradients['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(W.T, dZ)
        return gradients
    
    def update_parameters(self, gradients, learning_rate):
        for l in range(1, len(self.layer_dims)):
            self.parameters['W' + str(l)] -= learning_rate * gradients['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * gradients['db' + str(l)]
    
    def train(self, X_train, Y_train, num_epochs=1000, learning_rate=0.01):
        for epoch in range(num_epochs):
            Y_hat, caches = self.forward_propagation(X_train)
            loss = self.compute_loss(Y_hat, Y_train)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            gradients = self.backward_propagation(Y_hat, Y_train, caches)
            self.update_parameters(gradients, learning_rate)
    
    def predict(self, X_test):
        Y_hat, _ = self.forward_propagation(X_test)
        return Y_hat
