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

        self.K = len(np.unique(training_labels)) #typically 5

        # Multiple random initializations
        best_inertia = np.inf
        best_centroids = None
        best_cluster_assignments = None
        n_init = 10

        for init in range(n_init):
            # Randomly initialize centroids
            initial_indices = np.random.choice(training_data.shape[0], self.K, replace=False)
            centroids = training_data[initial_indices]

            for i in range(self.max_iters):
                # Assign clusters
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
                    centroids = new_centroids
                    break

            centroids = new_centroids

            inertia = np.sum((training_data - centroids[cluster_assignments])**2)

            # Keep the best clustering
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_cluster_assignments = cluster_assignments

        self.centroids = best_centroids
        pred_labels = best_cluster_assignments

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        
        distances = np.linalg.norm(test_data[:, np.newaxis] - self.centroids, axis=2)
        test_labels = np.argmin(distances, axis=1)

        return test_labels
