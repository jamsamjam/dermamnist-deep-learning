# Milestone 2 Report

362578 Clea Maisonnier | 375535 Sam Lee | 379094 Yahan Zhang

## Introduction
Our project goal is to diagnose skin lesions across seven categories, by classifying dermastopic images from the DermaMNIST dataset. We evaluate Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN).

## Method

### Multilayer Perceptron (MLP)
The MLP is a class of feed-forward artificial neural network that maps input data to appropriate outputs through multiple layers of neurons. It is consisted of :
- init: initializes the network. It defines the different layers such as the number and the size of each one. The size depends onthe differnt values of the hidden_layer.
-forward: performs the forward pass of the network. It applies linear transformation and the activation function.

### Convolutional Neural Network (CNN)


### Trainer

## Experiment/Results


| Method | Train Accuracy (%) | Train F1-score | Test Accuracy (%) | Test F1-score | Training Time (s) | Prediction Time (s) |
|:---|:---|:---|:---|:---|:---|:---|
| KNN | 64.979% | 0.359204 | 60.000% | 0.329790 | 0.0039 | 0.0009 |
| Logistic Regression | 64.979% | 0.386905 | 61.667% | 0.330476 | 0.0116 | 0.0000 |
| KMeans | 45.570% | 0.333175 | 21.667% | 0.260529 | 0.0290 | 0.0000 |

### Logistic Regression: Validation Accuracy vs Learning Rate:

To select the best learning rate, we performed hyperparameter tuning on both the learning rate and the number of maximum iterations simultaneously. 
We evaluated learning rates [1e-2, 1e-3, 5e-4, 1e-4] and max iterations [500, 1000, 2000, 5000].

Based on the validation set performance shown in the figure below, we selected `lr=1e-2` and `max_iters=500`, achieving the highest validation accuracy.

![](figures/validation_accuracy_lr.png){ width=60% }

### KNN: Which K to choose

We use cross-validation to find the K that gives the highest validation accuracy. Due to class imbalance in the dataset, accuracy alone may not be a reliable metric; thus, we also considered the Macro F1-Score, which better captures performance across all classes.
Although K=3 achieved the highest accuracy, K=7 provided the best F1-Score with comparable accuracy. Since K=7 also avoids ties, we selected K=7 as the final hyperparameter.

![](figures/validation_accuracy_k.png)

### Runtime Analysis

We measured the training and prediction times for each model once under the same hardware and software conditions. Given the small scale of the dataset and the fast execution times, single-run timing was considered sufficient.
KMeans required the longest training time due to its iterative update process, while KNN achieved instant training but showed slower prediction due to nearest neighbor searches. Logistic Regression achieved a good balance between training speed and generalization performance.

## Discussion/Conclusion

- **KMeans** showed limited performance despite leveraging true labels to select the best clustering. As clustering remained unsupervised, its classification performance was inherently limited, although some structure in the data was uncovered.
- **KNN** achieved high training accuracy but lower generalization to the test set.
- **Logistic Regression** provided reasonable generalization performance, with hyperparameter tuning — particularly the choice of learning rate and maximum iterations — having a significant impact.