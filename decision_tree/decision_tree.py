import numpy as np

class DecisionNode:
    def __init__(self, split_feature, split_value, left_subtree, right_subtree):
        self.split_feature = split_feature  
        self.split_value = split_value 
        self.left_subtree = left_subtree  
        self.right_subtree = right_subtree  
        
class LeafNode:
    def __init__(self, majority_label):
        self.majority_label = majority_label
        
class DecisionTree:
    def __init__(self):
        self.max_depth = 10
        self.min_sample_per_leaf = 5
    
    def calculate_mse(self, X, y, left_data, right_data):
        #we use mse determining the best split since it is easy to implement
        left_values = y[left_data]
        right_values = y[right_data]
        
        left_point = np.sum((left_values - np.mean(left_values)) ** 2)
        right_point = np.sum((right_values - np.mean(right_values)) ** 2)
        
        return left_point + right_point
    
    def find_best_split(self, X, y):
        """
        Evaluate all possible splits for each feature.
        Calculate impurity measures (e.g., Gini impurity, entropy) for each split.
        Select the split that maximally reduces impurity or maximizes information gain.
        """
        best_feature = None
        best_value = None
        best_point = -np.inf
        
        #we go through every feature and every value to find best split 
        for feature_number in range(X.shape[1]):
            feature_values = X[:, feature_number]
        
            #even though decision trees normally handle categorical values, just 
            #for simplicitys sake we assume feature values will be continuous
            #and we will take every value as potential split point
            for split_value in np.unique(feature_values):
                left_data = np.where(feature_values <= split_value)[0]
                right_data = np.where(feature_values > split_value)[0]
        
                #since point is mse we take the lowest as best
                point = self.calculate_mse(X, y, left_data, right_data)

                if point < best_point:
                    best_feature = feature_number
                    best_value = split_value
                    best_point = point
        
        #return value will be indexes of feature and value lists       
        return best_feature, best_value
    
    def split_data(self, X, y, split_feature, split_value):
        left_i = np.where(X[:, split_feature] <= split_value)[0]
        right_i = np.where(X[:, split_feature] > split_value)[0]
        
        left_X, left_y = X[left_i], y[left_i]
        right_X, right_y = X[right_i], y[right_i]
        
        return left_X, left_y, right_X, right_y
    
    def stopping_criteria(self, X, y, depth):
        if not (depth < self.max_depth) or not (len(y) < self.min_sample_per_leaf):
            return True
        return False
        #TODO add minimum impurity decrease or minimum improvement cases
        
    def most_common_class(self, y):
        #returns most common label in a subset
        unique_items, counts = np.unique(y, return_counts=True)
        return unique_items[np.argmax(counts)]
        
    def build_tree(self, X, y, depth):
        #check if a stopping criteria is met
        #return node as a leaf node(terminal node)
        if self.stopping_criteria(X, y, depth):
            return LeafNode(self.most_common_class(y))
        
        #find best split using gini/purity vs.
        split_feature, split_value = self.find_best_split(X, y)
        
        if split_feature is None:
            return LeafNode(self.most_common_class(y))
        
        left_X, left_y, right_X, right_y = self.split_data(X, y, split_feature, split_value)
        
        left_subtree = self.build_tree(left_X, left_y, depth + 1)
        right_subtree = self.build_tree(right_X, right_y, depth + 1)
        
        #create new decision node
        #(since first decision node (root node) will be returned at last,
        #this function will return root node)
        return DecisionNode(split_feature, split_value, left_subtree, right_subtree)
    
    
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=15, n_classes=3, 
    n_clusters_per_class=1, weights=[0.5, 0.5], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTree()
dt.build_tree(X_train, y_train, 0)