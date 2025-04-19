import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)

def train_split(X, y, test_size=0.2):
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data

    # EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load("features.npz", allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"].astype(int), feature_data["ytest"].astype(int) # TODO

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
         #Split data into 80% training and 20% validation sets
        xtrain, xtest, ytrain, ytest = train_split(xtrain, ytrain, test_size=0.2)

    #Normalize data
    xtest = normalize_fn(xtest, np.mean(xtrain, 0, keepdims=True), np.std(xtrain, 0, keepdims=True))
    xtrain = normalize_fn(xtrain, np.mean(xtrain, 0, keepdims=True), np.std(xtrain, 0, keepdims=True))
    

    # Append bias term for logistic regression
    if args.method == "logistic_regression":
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr = args.lr, max_iters = args.max_iters)
    elif args.method == "kmeans":
        method_obj = KMeans(max_iters=args.max_iters)
    elif args.method == "knn":
        method_obj = KNN(k = args.K)
    
    ## 4. Train and evaluate the method
    if args.method == "logistic_regression" and not args.test:
        learning_rates = [1e-2, 1e-3, 5e-4, 1e-4]
        max_iters_list = [500, 1000, 2000, 5000]

        results = {}

        for lr in learning_rates:
            accs = []
            for max_iters in max_iters_list:
                print(f"Training Logistic Regression with lr={lr}, max_iters={max_iters}")
                model = LogisticRegression(lr=lr, max_iters=max_iters)
                model.fit(xtrain, ytrain)
                preds_val = model.predict(xtest)
                val_acc = accuracy_fn(preds_val, ytest)
                accs.append(val_acc)
            results[lr] = accs

        plt.figure(figsize=(10,7))
        for lr in learning_rates:
            plt.plot(max_iters_list, results[lr], marker='o', label=f'lr={lr}')

        plt.xlabel('Max Iterations')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs Max Iterations for Different Learning Rates')
        plt.legend()
        plt.grid(True)
        plt.savefig('figures/validation_accuracy_lr.png')
        plt.show()

    else:
        preds_train = method_obj.fit(xtrain, ytrain)

        preds = method_obj.predict(xtest)

        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    
    #Visualizations for KNN
    
    # K range for accuracy and F1 scores
    if args.method == "knn":
        k_values = range(1, 21)
        odd_k_values = [k for k in k_values if k % 2 != 0]

        # Accuracy (Training vs Validation)
        train_errors_acc = []
        val_errors_acc = []

        for k in k_values:
            knn = KNN(k=k)
            knn.fit(xtrain, ytrain)

            ypred_train = knn.predict(xtrain)
            train_error = accuracy_fn(ypred_train, ytrain)
            train_errors_acc.append(train_error)

            ypred_val = knn.predict(xtest)
            val_error = accuracy_fn(ypred_val, ytest)
            val_errors_acc.append(val_error)

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, train_errors_acc, 'r', label='Training Accuracy')
        plt.plot(k_values, val_errors_acc, 'g', label='Validation Accuracy')

        # Find best odd k 
        best_odd_acc_k = odd_k_values[np.argmax([val_errors_acc[k-1] for k in odd_k_values])]
        plt.axvline(x=best_odd_acc_k, color='purple', linestyle=':', 
                    label=f'Best odd k (k={best_odd_acc_k})')
        # Find best overall k 
        best_acc_k = k_values[np.argmax([val_errors_acc[k-1] for k in k_values])]
        plt.axvline(x=best_acc_k, color='pink', linestyle=':', 
                    label=f'Best overall k (k={best_acc_k})')

        plt.xlabel('Number of Neighbors (k)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Training vs Validation Accuracy', fontsize=14)
        plt.xticks(k_values)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # F1 Score (Training vs Validation)
        train_errors_f1 = []
        val_errors_f1 = []

        for k in k_values:
            knn = KNN(k=k)
            knn.fit(xtrain, ytrain)

            ypred_train = knn.predict(xtrain)
            train_errors_f1.append(macrof1_fn(ypred_train, ytrain))

            ypred_val = knn.predict(xtest)
            val_errors_f1.append(macrof1_fn(ypred_val, ytest))

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, train_errors_f1, 'r', label='Training F1 Score')
        plt.plot(k_values, val_errors_f1, 'g', label='Validation F1 Score')

        # Find best odd k 
        best_odd_f1_k = odd_k_values[np.argmax([val_errors_f1[k-1] for k in odd_k_values])]
        plt.axvline(x=best_odd_f1_k, color='purple', linestyle=':', 
                    label=f'Best odd k (k={best_odd_f1_k})')
        # Find best overall k 
        best_f1_k = k_values[np.argmax([val_errors_f1[k-1] for k in k_values])]
        plt.axvline(x=best_f1_k, color='pink', linestyle=':', 
                    label=f'Best overall k (k={best_f1_k})')

        plt.xlabel('Number of Neighbors (k)', fontsize=12)
        plt.ylabel('F1 score', fontsize=12)
        plt.title('Training vs Validation F1 Score', fontsize=14)
        plt.xticks(k_values)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    



if __name__ == "__main__":
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)",
    )
    parser.add_argument(
        "--data_path", default="data", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="features/original(MS2)"
    )
    parser.add_argument(
        "--K", type=int, default=1, help="number of neighboring datapoints used for knn"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument(
        "--nn_type",
        default="cnn",
        help="which network to use, can be 'Transformer' or 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
