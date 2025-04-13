## Introduction

## Method

## Experiment/Results

## Discussion/Conclusion


###################### Updated for KMeans #####################
### KMens Clustering
We implemented KMeans with K=5 (number of classes). As an unsupervised method, KMeans does not use label information during training. Therefore, the classification performance is limited when evaluated using accuracy or F1 score.

Our implementation achieved:

- Train accuracy: 20.7%, F1: 0.18  
- Test accuracy: 15.0%, F1: 0.14

These values are expected due to the nature of unsupervised learning. 
Nonetheless, the model is able to discover some structure in the data space.