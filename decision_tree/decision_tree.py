import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class Node:
    def __init__(self, parent = None, children = None, decision = None, child_var = None, variable = None, *, value=None):
        self.children = children # A dictionary with category names as keys and child nodes as value
        self.decision = decision # A response category. I only not None if the node is a leaf.
        self.child_var = child_var # The variable name for which the child nodes are made form
    
    def is_leaf(self):
        return self.decision is not None

    def set_children(self, children):
        self.children = children
    
    def set_decision(self, decision):
        self.decision = decision
    
    def set_child_var(self, child_var):
        self.child_var = child_var
    
    def get_children(self):
        return self.children
    
    def get_child_var(self):
        return self.child_var
    
    def get_decision(self):
        return self.decision
    


class DecisionTree:
    
    def __init__(self, max_depth = np.float("inf")):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.root = None
        self.rules = []
        self.response = None
        self.max_depth = max_depth
    
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.response = np.unique(y) # Saves the response variable
        self.root = self.make_tree(X, y, 0) # Returns the root of the decision tree

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        y_hat = []
        # For each element in the data X, the tree is traversed to find the decsion provided på the tree
        for index, row in X.iterrows():
            y_hat.append(self.traverse_tree(row, self.root))
        
        return np.array(y_hat)
    
        
    def traverse_tree(self, x, node):
        """
        Traverses the tree until a node with a decision is found and returns that decision.
        Args:
            x (pd.DataFrame.series): A single row of the DataFrame X
            node (decision_tree.Node): A node in the decision tree 
        
        Returns: 
            A length m vector with predictions
        """
        if node.is_leaf():
            return node.get_decision()

        child_var = node.get_child_var() 

        category = x[child_var]
        try:
            child = node.get_children()[category]
        except:
            return np.random.choice(self.response)
        return self.traverse_tree(x, child)

    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model1.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # If rules is not empty
        if self.rules:
            return self.rules
        
        # If rule list is empty, make a new rule list
        self.make_rules(self.root, [])
        return self.rules 


    def make_rules(self, node, rule_list):
        """
        Makes a list of rules by traversing the tree and saving the decisions on the on the desired form. 
        Args: 
            node (decision_tree.Node): A node in the decision tree
            rule_list: A list of tuples, where each tuple contains variable and it's corresponding category that has been traversed to get to the current node
        """
        for child_name, child_node in node.get_children().items():
            rule = (node.get_child_var(), child_name)
            current_rule_list = rule_list.copy()
            current_rule_list.append(rule)
            if child_node.is_leaf():
                decision = (current_rule_list, child_node.get_decision())
                self.rules.append(decision)
            
            else:
                self.make_rules(child_node, current_rule_list)
    
    def get_root(self):
        return self.root

    def make_tree(self, X, y, depth):
        """
        The algorith to make the tree. 
        """
        current_node = Node()
        count, response_list = y_count(X, y)
        variable_list = np.unique(X.columns)
        # Check if we should return a node with a decicion
        if entropy(count) == 0 or len(variable_list) == 1 or depth == self.max_depth:
            count = list(count)
            max_index = count.index(max(count))
            current_node.set_decision(response_list[max_index])
            return current_node

        # If no decision can be made, we need to continue building the tree
        max_gain = -1
        best_var = None

        for var in variable_list:
            gain = info_gain(X, y, var)
            if gain>max_gain:
                max_gain = gain
                best_var = var
        
        current_node.set_child_var(best_var)
        category_list = np.unique(X[best_var])
        children = {}
        for cat in category_list:
            ind = X[best_var] == cat
            X_sub, y_sub = X[ind], y[ind]
            del X_sub[best_var]

            child_node = self.make_tree(X_sub, y_sub, depth + 1)

            children[cat] = child_node
        current_node.set_children(children)

        return current_node



    


# --- Some utility functions 

    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))

def info_gain(X, y, var_name):
    x = X[var_name]
    
    entropies = {}
    vars = np.unique(x)
    resps = np.unique(y)

    count = np.zeros(len(resps))
    for resp in y:
        count = count + [resps == resp]
    gain = entropy(count)

    for var in vars:
        count = np.zeros(len(resps))
        y_sub = y[x == var]
        
        for resp in y_sub:
            count = count + [resps == resp]
        
        # Save entropy and number of elements in category
        entropies[var] = [entropy(count), len(y_sub)]
    
    for var in vars:
        entropy_list = entropies[var]
        gain -= (entropy_list[1]/len(y))*entropy_list[0]
    
    return gain 

def y_count(X, y):
    response_list = np.unique(y)
    count = np.zeros(len(response_list))

    for response in response_list:
        count = count + [response_list == response]
    
    return count, response_list