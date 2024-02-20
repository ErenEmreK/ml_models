import numpy as np
from math import e, log
"""
    Example Usage:
    
    #Create an example dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, 
        n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LogisticRegression(len(X_train[0]))
    lr.train(X_train, y_train, 10000, 0.01)
    lr.test(X_test, y_test) 

"""



class LogisticRegression:
    def __init__(self, feature_count):
        self.weights, self.bias = self.initialize_parameters(feature_count)
           
    def initialize_parameters(self, feature_count):
        # Initialize weights and bias randomly
        weights = np.random.randn(feature_count)
        bias = np.random.randn()
        
        return weights, bias
    
    def forward_propagation(self, X, weights, bias):
        #we use sigmoid function to fit our guess into binary categorization
        z = np.dot(X, weights) + bias
        sigmoid = 1 / (1 + e ** -z) 
        
        return sigmoid
    
    def loss(self, predicted_probs, true_labels):
        epsilon = 1e-10
        predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    
        #calculate loss with binary cross-entropy
        return -np.mean(true_labels * np.log(predicted_probs) + (1 - true_labels) * np.log(1 - predicted_probs))
        
   

    def backpropagation(self, X, predicted_probs, true_labels):
        #it's same backpropagation func from our linear regression model
        
        #gradients w.r.t predictions
        d_loss_prediction = predicted_probs - true_labels

        #gradients of loss w.r.t weights and bias using chain rule
        d_weights = np.dot(X.T, d_loss_prediction) / len(true_labels)
        d_bias = np.mean(d_loss_prediction)
        
        return d_weights, d_bias

    def update_parameters(self, weights, bias, dw, db, learning_rate):
        #update weight and bias regarding learning rate and gradients
        weights -= learning_rate * dw
        bias -= learning_rate * db

        return weights, bias
    
    def train(self, X_train, y_train, epochs, learning_rate):
        weights, bias = self.weights, self.bias
        
        for epoch in range(epochs):
            
            predictions = self.forward_propagation(X_train, weights, bias)
            loss = self.loss(predictions, y_train)
            
            print(f"Epoch: {epoch}, Loss: {loss}")
            
            dw, db = self.backpropagation(X_train, predictions, y_train)
            weights, bias = self.update_parameters(weights, bias, dw, db, learning_rate)
            
        self.weights, self.bias = weights, bias
        return weights, bias
    
    def test(self, X_test, y_test): 
        predictions = self.forward_propagation(X_test, self.weights, self.bias)
        binary_predictions = np.round(predictions)
        
        accuracy = np.mean(binary_predictions == y_test)
        
        print(f"Accuracy: {accuracy * 100}%")
        
        return accuracy
    
