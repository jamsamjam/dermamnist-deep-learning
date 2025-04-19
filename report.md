# Milestone 1 Report

375535 Sam Lee | 362578 | 379094

## Introduction

## Method

### K-Nearest Neighbors (KNN)

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

## Discussion/Conclusion

- **KMeans** showed low performance as expected for unsupervised clustering.
- **KNN** achieved high training accuracy but lower generalization to the test set.
- **Logistic Regression** provided reasonable generalization performance after hyperparameter tuning.
- Hyperparameter tuning, especially learning rate selection, had a significant impact on Logistic Regression performance.

