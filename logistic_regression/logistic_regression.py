import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, learning_rate = 0.01, max_iter = 1000, tol = 1e-12, weigths = None, intercept = 0, non_linear = False, degree = 1):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weigths = weigths
        self.intercept = intercept
        self.non_linear = non_linear
        self.degree = degree
    
    def preprocess(self, data):
        X = data.copy()
        n_cols = X.shape[1]
        for d in range(2, self.degree+1):
            for i in range(n_cols):
                col = X.columns[i]
                X[str(col)+"^"+str(d)] = X[col]**d
            
        X = np.c_[np.ones(X.shape[0]), X]
        return X
    
    def fit(self, data, y):
        """
        Estimates parameters for the classifier
        
        Args:
            data (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        # Transform matrix
        X = self.preprocess(data)
        # Number of variables
        k = X.shape[1]
        # Initiate beta
        beta_0 = np.repeat(0,k)
        
        for i in range(self.max_iter):
            nu = np.matmul(X,beta_0)
            p = sigmoid(nu)
            
            grad = np.matmul( np.transpose(X),(y-p))
            W = np.diag(p*( np.repeat(1,len(p)) - p))
            inf_mat = np.matmul(np.transpose(X), np.matmul(W, X))
            inv_mat = np.linalg.inv(np.linalg.cholesky(inf_mat))
            
            beta_1 = beta_0 + self.learning_rate*np.matmul(inv_mat, grad)
            if min(abs((beta_1-beta_0)/beta_0)) < self.tol:
                break
            beta_0 = beta_1
        self.weights = beta_1
        
        
    
    def predict(self, data):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            data (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        X = self.preprocess(data)
        
        return sigmoid(np.matmul(X,self.weights))

        
# --- Some utility functions 


def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))