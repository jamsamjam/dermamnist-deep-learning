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
        xtrain, xtest, ytrain, ytest = train_split(xtrain, ytrain, test_size=0.2)

    xtrain = normalize_fn(xtrain, np.mean(xtrain, axis=0), np.std(xtrain, axis=0))
    xtest = normalize_fn(xtest, np.mean(xtrain, axis=0), np.std(xtrain, axis=0))

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
        val_accuracies = []

        for lr in learning_rates:
            print(f"\nTraining Logistic Regression with lr = {lr}")

            model = LogisticRegression(lr=lr, max_iters=args.max_iters)

            model.fit(xtrain, ytrain)
            preds_val = model.predict(xtest)

            val_acc = accuracy_fn(preds_val, ytest)
            val_accuracies.append(val_acc)

            print(f"Validation set: accuracy = {val_acc:.3f}")

        best_idx = np.argmax(val_accuracies)
        best_lr = learning_rates[best_idx]
        print(f"\nBest learning rate selected: {best_lr}")

        plt.figure(figsize=(8,6))
        plt.plot(learning_rates, val_accuracies, marker='o')
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs Learning Rate')
        plt.grid(True)
        plt.savefig('logistic_regression_lr_tuning.png')
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
