import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


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

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.k = label_to_onehot(training_labels).shape[1]
        self._logistic_regression_train_multi(training_data, training_labels)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        return self._logistic_regression_classify_multi(test_data)
            
    def _logistic_regression_train_multi(self, X, y):
        self.w = np.random.normal(0., 0.1, (X.shape[1], self.k))
        Y = label_to_onehot(y)

        for epoch in range(self.max_iters):
            gradient = self._gradient_logistic_multi(X, Y)
            self.w -= self.lr * gradient

            y_pred = self._logistic_regression_classify_multi(X)
            if self._accuracy(y, y_pred) >= 0.999:
                break   
    
    def _gradient_logistic_multi(self, X, Y):
        grad_w = X.T @ (self._softmax(X @ self.w) - Y)
        return grad_w

    def _accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def _logistic_regression_classify_multi(self, X):
        logits = X @ self.w
        return np.argmax(self._softmax(logits), axis=1)
    
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _sigmoid(self, t):
        return 1 / (1 + np.exp(-np.clip(t, -500, 500)))
    