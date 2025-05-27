import numpy as np

from ..utils import label_to_onehot, append_bias_term, get_n_classes

class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.mean = None
        self.std = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.k = get_n_classes(training_labels)

        # Normalize and add bias term
        self._normalize_fit(training_data)
        X = self._normalize_transform(training_data)
        X = append_bias_term(X)

        N, D = X.shape
        np.random.seed(42)
        self.weights = np.random.normal(0., 0.1, (D, self.k))
        Y = label_to_onehot(training_labels)

        for _ in range(self.max_iters):
            gradient = self._compute_gradient(X, Y) / N
            self.weights -= self.lr * gradient

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        X = self._normalize_transform(test_data)
        X = append_bias_term(X)
        logits = X @ self.weights
        return np.argmax(self._softmax(logits), axis=1)
    
    def _normalize_fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8

    def _normalize_transform(self, X):
        return (X - self.mean) / self.std
    
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_gradient(self, X, Y):
        predictions = self._softmax(X @ self.weights)
        return X.T @ (predictions - Y)