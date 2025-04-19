import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels= training_labels
        return training_labels
    
    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        predictions = []
        for val in test_data:
            predictions.append(self._kNN_predict_one(val))
        test_labels = np.array(predictions)
        
        return test_labels
    
    def _euclidean_dist(self,example):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.
        """
        sum = np.sum((self.training_data - example)**2, axis=1)
        return np.sqrt(sum)

    def _find_k_nearest_neighbors(self, distances):
        """ Find the indices of the k smallest distances from a list of distances.
        """
        sorted_indices = np.argsort(distances)
        return sorted_indices[:self.k]

    def _predict_label(self,neighbor_labels):
        """Return the most frequent label in the neighbors'.
        """
        counts = np.bincount(neighbor_labels.astype(int))
        return np.argmax(counts)

    def _kNN_predict_one(self,unlabeled_example):
        """Returns the label of a single unlabelled example.
        """
        distances = self._euclidean_dist(unlabeled_example)
        nn_indices = self._find_k_nearest_neighbors(distances)
        neighbor_labels = self.training_labels[nn_indices]
        best_label = self._predict_label(neighbor_labels) 

        return best_label
