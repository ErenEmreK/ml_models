import numpy as np

class DecisionTree:
    def __init__(self):
        pass
    
    def find_best_split(self):
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
    
    def stopping_criteria(self, data, max_depth):
        pass
        
    def build_tree(self, data, target, max_depth):
        
        if self.stopping_criteria(data, max_depth):
            return self.create_leaf_node()
        
        split_feature = self.find_best_split()
        left, right = self.split_data(split_feature)
        
        