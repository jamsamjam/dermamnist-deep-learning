# Milestone 1 Report

375535 Sam Lee | 362578 | 379094

## Introduction
Our project goal is to predict the presence of heart disease in a patient, categorized on a scale from 0 to 4.
In this first part, we will utilize three classifiers: logistic regression, KNN and K-means.

## Method

### K-Nearest Neighbors (KNN)
The KNN is an algorithm which classifies a given argument x according to most of its k nearest neighbors in the training set. 
- init: assigns to the current object the given argument k (by default k=1),and the task's type
- fit: stores the current object the training data as well as its labels,returns the predicted labels
- predict:classifies each point in the data test depending on the majority label of its nearest k neighbors, using the Euclidean distance. 

### Logistic Regression
We implemented Logistic Regression using gradient descent optimization.  
To select the best learning rate, we performed hyperparameter tuning based on validation set performance.

We tested the following learning rates: `1e-2`, `1e-3`, `5e-4`, `1e-4`.  
As shown in Figure 1 below, the validation accuracy dropped significantly when the learning rate was `1e-2`.

Based on the results, we selected:
- Best learning rate: `1e-3`
- Max iterations: `1000`

### KMeans Clustering
We implemented KMeans with K=5 (number of classes). As an unsupervised method, KMeans does not use label information during training. Therefore, the classification performance is limited when evaluated using accuracy or F1 score.

Our implementation achieved:

- Train accuracy: 20.7%, F1: 0.18  
- Test accuracy: 15.0%, F1: 0.14

These values are expected due to the nature of unsupervised learning. 
Nonetheless, the model is able to discover some structure in the data space.

## Experiment/Results

| Method | Train Accuracy | Test Accuracy | Test F1-score |
|:---|:---|:---|:---|
| KNN (k=?) | 100% | XX% | XX |
| Logistic Regression (lr=1e-3) | 60% | 44% | 0.16 |
| KMeans | XX | XX% | XX |

**Validation Accuracy vs Learning Rate (Logistic Regression):**

![Validation Accuracy vs Learning Rate](figures/lr_tuning.png)

*(Figure 1: Validation set accuracy for different learning rates.)*
### K-Nearest Neighbors (KNN): Which K to choose

We use the Cross Validation method, which consists of finding the K that gives us the highest validation accuracy. In this dataset, there is a class imbalance (the size of class 0 in the training set is 128, for class 4 it is 8) and since Accuracy can be biased towards the largest class, we must consider the F1 Score. Also, K is usually chosen as an odd number to avoid ties in classification. From the validation results, we can see that the odd value that yields to the highest accuracy is 3, while for the F1 score it’s 7. Since considering F1 is important and K=7 gives us a similar rate for accuracy as K=3, hence we chose K=7 as the final hyperparameter.
(figures/validation_accuracy.png)
*(Figure 2: Validation accuracy for different K values.)*
*(Figure 3: Validation F1 score for different K values.)*

## Discussion/Conclusion

- **KMeans** showed low performance as expected for unsupervised clustering.
- **KNN** achieved high training accuracy but lower generalization to the test set.
- **Logistic Regression** provided reasonable generalization performance after hyperparameter tuning.
- Hyperparameter tuning, especially learning rate selection, had a significant impact on Logistic Regression performance.

