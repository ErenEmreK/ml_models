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
        pass
    
    def find_best_split(self, X, y):
        """
        Evaluate all possible splits for each feature.
        Calculate impurity measures (e.g., Gini impurity, entropy) for each split.
        Select the split that maximally reduces impurity or maximizes information gain.
        """
        pass
    
    def split_data(self):
        
        pass
    
    def create_leaf_node(self):
        pass
    
    def stopping_criteria(self, X, y, max_depth):
        pass
    
    def most_common_class(self, y):
        pass
        
    def build_tree(self, X, y, target, max_depth):
        #check if a stopping criteria is met
        #return node as a leaf node(terminal node)
        if self.stopping_criteria(X, y, max_depth):
            return LeafNode(self.most_common_class(y))
        
        #find best split using gini/purity vs.
        split_feature, split_value = self.find_best_split(X, y)
        
        if split_feature is None:
            return LeafNode(self.most_common_class(y))
        
        left_X, left_y, right_X, right_y = self.split_data(split_feature)
        
        left_subtree = self.build_tree(left_X, left_y, target, max_depth - 1)
        right_subtree = self.build_tree(right_X, right_y, target, max_depth - 1)
        
        #create new decision node
        #(since first decision node (root node) will be returned at last,
        #this function will return root node)
        return DecisionNode(split_feature, split_value, left_subtree, right_subtree)
    
    