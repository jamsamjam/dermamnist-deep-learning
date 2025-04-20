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
        n_init = 10
        best_accuracy = 0
        best_centroids = None
        best_cluster_assignments = None

        for init in range(n_init):
            # Randomly initialize centroids
            initial_indices = np.random.choice(training_data.shape[0], self.K, replace=False)
            centroids = training_data[initial_indices]

            for i in range(self.max_iters):
                # Assign clusters
                distances = np.linalg.norm(training_data[:, np.newaxis] - centroids, axis=2)
                cluster_assignments = np.argmin(distances, axis=1)

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

            best_match_acc, best_mapped_labels = self._best_cluster_matching(cluster_assignments, training_labels)

            if best_match_acc > best_accuracy:
                best_accuracy = best_match_acc
                best_centroids = centroids
                best_cluster_assignments = best_mapped_labels

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
    
    def _best_cluster_matching(self, pred_labels, true_labels):
        import itertools
        best_acc = 0
        best_mapping = None
        K = len(np.unique(true_labels))

        for permutation in itertools.permutations(range(K)):
            mapped = np.array([permutation[label] for label in pred_labels])
            acc = np.mean(mapped == true_labels)
            if acc > best_acc:
                best_acc = acc
                best_mapping = permutation

        best_mapped_labels = np.array([best_mapping[label] for label in pred_labels])

        return best_acc, best_mapped_labels
        