import numpy as np
"""
    Example Usage:
    X_train = np.array([[1, 3, 2], [1, 2, 3], [1, 3, 4]])
    y_train = np.array([3, 4, 5])
    X_test = np.array([[4, 5, 5], [1, 5, 6]])
    y_test = np.array([6, 7])
    epochs = 1000
    learning_rate = 0.01
    
    lr = LinearRegression(len(X_train[1]))
    lr.train(X_train, y_train, epochs, learning_rate)
    lr.test(X_test, y_test)
"""
    
class LinearRegression:
    def __init__(self, feature_count):
        self.weights, self.bias = self.initialize_parameters(feature_count)
           
    def initialize_parameters(self, feature_count):
        #we initialize weights and bias randomly
        weights = np.random.randn(feature_count)
        bias = np.random.randn()
        
        return weights, bias

    def forward_propagation(self, X, weights, bias):
        #linear regression function
        return np.dot(X, weights) + bias

    def loss(self, predicted_labels, true_labels):
        #calculate loss with mean squared error
        return np.mean((predicted_labels - true_labels) ** 2)

    def backpropagation(self, X, predicted_labels, true_labels):
        #gradients w.r.t predictions
        d_loss_prediction = predicted_labels - true_labels

        #since activation func not exists in linear reg.
        #derivative of weighted sum w.r.t itself will be 1
        d_loss_z = d_loss_prediction * 1

        #gradients of loss w.r.t weights and bias using chain rule
        d_weights = np.dot(X.T, d_loss_z) / len(true_labels)
        d_bias = np.mean(d_loss_z)
        
        return d_weights, d_bias
    
    def update_parameters(self, weights, bias, dw, db, learning_rate):
        #update weight and bias regarding learning rate and gradients
        weights -= learning_rate * dw
        bias -= learning_rate * db

        return weights, bias

    def train(self, X_train, y_train, epochs=100, learning_rate=0.01):
        weights, bias = self.initialize_parameters(len(X_train[1]))
        
        for epoch in range(epochs):
            predictions = self.forward_propagation(X_train, weights, bias)
            loss = self.loss(predictions, y_train)
            
            print(f"Epoch {epoch}, Loss: {loss} ")
                
            dw, db = self.backpropagation(X_train, predictions, y_train)
            weights, bias = self.update_parameters(weights, bias, dw, db, learning_rate)
            self.weights, self.bias = weights, bias
            
        return weights, bias

    def test(self, X_test, y_test):
        predictions = self.forward_propagation(X_test, self.weights, self.bias)
        
        test_loss = self.loss(predictions, y_test)
        
        print(f"Test loss: {test_loss}")
        
        return test_loss
        