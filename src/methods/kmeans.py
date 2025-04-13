import numpy as np
import itertools


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, max_iters=500):
        """
        Call set_arguments function of this class.
        """
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        ###################### UPDATE1 BEGIN ############################

        self.K = len(np.unique(training_labels)) #typically 5

        #initialize centroids
        initial_indices = np.random.choice(training_data.shape[0], self.K, replace = False)
        centroids = training_data[initial_indices]

        for i in range(self.max_iters):
            #assign clusters
            distance = np.linalg.norm(training_data[:, np.newaxis] - centroids, axis = 2)
            cluster_assignments = np.argmin(distance, axis=1)
            
            # Compute new centroids
            new_centroids = np.array([
                training_data[cluster_assignments == k].mean(axis=0)
                if np.any(cluster_assignments == k) else centroids[k]
                for k in range(self.K)
            ])

            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        self.centroids = centroids
        pred_labels = cluster_assignments

        ###################### UPDATE1 ENDS ############################

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        
        ###################### UPDATE2 BEGIN ############################
        distances = np.linalg.norm(test_data[:, np.newaxis] - self.centroids, axis=2)
        test_labels = np.argmin(distances, axis=1)
        ###################### UPDATE2 ENDS ############################

        return test_labels
