import numpy as np
"""
    Example Usage:
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=15, n_classes=3, 
        n_clusters_per_class=1, weights=[0.3, 0.3, 0.4], random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    dt = DecisionTree(split_parameter=calculate_mse)
    dt.create_tree(X_train, y_train)
    dt.test(X_test, y_test)
    
"""
def calculate_mse(X, y, left_data, right_data):
    #we use mse determining the best split since it is easy to implement
    left_values = y[left_data]
    right_values = y[right_data]
    
    left_point = np.sum((left_values - np.mean(left_values)) ** 2)
    right_point = np.sum((right_values - np.mean(right_values)) ** 2)
    
    return left_point + right_point
    
def gini_impurity(X, y, left_data, right_data):
    # Get the labels for left and right subsets
    left_labels = y[left_data]
    right_labels = y[right_data]
    
    # Calculate the number of samples in each subset
    total_samples = len(left_labels) + len(right_labels)
    
    # Calculate the proportion of each class in left and right subsets
    left_counts = np.bincount(left_labels)
    right_counts = np.bincount(right_labels)
    
    # Calculate the Gini impurity for left and right subsets
    left_gini = 1 - np.sum((left_counts / len(left_labels))**2)
    right_gini = 1 - np.sum((right_counts / len(right_labels))**2)
    
    # Calculate the weighted average Gini impurity
    gini_impurity = (len(left_labels) / total_samples) * left_gini + (len(right_labels) / total_samples) * right_gini
    
    return gini_impurity

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
    def __init__(self, max_depth=5, min_sample_per_leaf=20, split_parameter=calculate_mse):
        self.max_depth = max_depth
        self.min_sample_per_leaf = min_sample_per_leaf
        self.split_parameter = split_parameter
        self.tree = None
        
    def find_best_split(self, X, y):
        """
        Evaluate all possible splits for each feature.
        Calculate impurity measures (e.g., Gini impurity, entropy) for each split.
        Select the split that maximally reduces impurity or maximizes information gain.
        """
        best_feature = None
        best_value = None
        best_point = np.inf
        
        #we go through every feature and every value to find best split 
        for feature_number in range(X.shape[1]):
            feature_values = X[:, feature_number]
        
            #even though decision trees normally handle categorical values, just 
            #for simplicitys sake we assume feature values will be continuous
            #and we will take every value as potential split point
            for split_value in np.unique(feature_values):
                left_data = np.where(feature_values <= split_value)[0]
                right_data = np.where(feature_values > split_value)[0]
        
                #we get lowest impurity or mse 
                point = self.split_parameter(X, y, left_data, right_data)

                if point < best_point:
                    best_feature = feature_number
                    best_value = split_value
                    best_point = point
        
        #return value will be indexes of feature and value lists       
        return best_feature, best_value
    
    def split_data(self, X, y, split_feature, split_value):
        left = np.where(X[:, split_feature] <= split_value)[0]
        right = np.where(X[:, split_feature] > split_value)[0]
        
        #check if one of subtrees are empty
        if len(left) == 0 or len(right) == 0:
            return None, None, None, None
        
        left_X, left_y = X[left], y[left]
        right_X, right_y = X[right], y[right]
        
        return left_X, left_y, right_X, right_y
    
    def stopping_criteria(self, X, y, depth):
        if not (depth < self.max_depth) or (len(y) < self.min_sample_per_leaf) or not (X.size and y.size):
            return True
        return False
        #TODO add minimum impurity decrease or minimum improvement cases
        
    def most_common_class(self, y):
        #returns most common label in a subset
        unique_items, counts = np.unique(y, return_counts=True)
        return unique_items[np.argmax(counts)]
        
    def build_tree(self, X, y, depth=0):
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
    
    def create_tree(self, X, y):
        self.tree = self.build_tree(X, y)
        print("Decision Tree has been built.")
    
    def get_label(self, instance):
        #we return label for instance(one data of a X dataset) 
        node = self.tree
        while not isinstance(node, LeafNode):
            value = instance[node.split_feature]
            node = node.left_subtree if value <= node.split_value else node.right_subtree
        return node.majority_label
        
    def test(self, X, y):
        if self.tree:
            predictions = np.array([self.get_label(i) for i in X])
            y = np.array(y)
            
            if len(y) == len(predictions):
                accuracy = np.sum(predictions == y) / len(y)
                print(f"Test accuracy: {accuracy * 100}%")
                
                return accuracy
        
            print("Prediction list must have same length as label list.")
            return None
        
        print("There is no available tree.")
        return None
    
