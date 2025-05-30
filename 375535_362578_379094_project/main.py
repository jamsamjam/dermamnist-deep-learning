import argparse

import numpy as np
from torchinfo import summary
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, plot_confusion_matrix


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, y_test = load_data()

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Make a validation set
    if not args.test:
        N = xtrain.shape[0]
        split = int(0.8 * N)
        perm = np.random.permutation(N)
        idx_train, idx_val = perm[:split], perm[split:]

        xval = xtrain[idx_val]
        yval = ytrain[idx_val]
        xtrain = xtrain[idx_train]
        ytrain = ytrain[idx_train]

    xtrain = xtrain / 255.0
    xtest = xtest / 255.0
    if not args.test:
        xval = xval / 255.0

    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)

    if args.nn_type == "mlp":
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
        if not args.test:
            xval = xval.reshape(xval.shape[0], -1)
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes)
    
    elif args.nn_type == "cnn":
        xtrain = xtrain.transpose(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
        xtest = xtest.transpose(0, 3, 1, 2)
        if not args.test:
            xval = xval.transpose(0, 3, 1, 2)
        model = CNN(input_channels=3, n_classes=n_classes)

    summary(model)

    # Trainer object
    method_obj = Trainer(model,
                        lr=args.lr,
                        epochs=args.max_iters,
                        batch_size=args.nn_batch_size,
                        device=args.device,
                        early_stop_patience=5,
                        xval=xval,
                        yval=yval)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    # preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.

    if args.test:
        preds = method_obj.predict(xtest)
        acc = accuracy_fn(preds, y_test)
        macrof1 = macrof1_fn(preds, y_test)
        print(f"Test set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
    else:
        preds = method_obj.predict(xval)
        acc = accuracy_fn(preds, yval)
        macrof1 = macrof1_fn(preds, yval)
        print(f"Validation set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        num_classes = get_n_classes(yval)
        plot_confusion_matrix(yval, preds, num_classes, normalize=True)

    ## Different learning rates and max_iters
    results = []

    learning_rates = [1e-3, 1e-4, 1e-5]
    max_iters_list = [10, 50, 100, 500]

    print("\n--- Hyperparameter sweep (lr × max_iters) ---")
    for lr in learning_rates:
        for max_iters in max_iters_list:
            print(f"\n>> Training with lr={lr}, max_iters={max_iters}")

            # Re-initialize the model
            if args.nn_type == "mlp":
                model = MLP(input_size=xtrain.shape[1], n_classes=n_classes)
            elif args.nn_type == "cnn":
                model = CNN(input_channels=3, n_classes=n_classes)

            trainer = Trainer(model,
                                lr=args.lr,
                                epochs=args.max_iters,
                                batch_size=args.nn_batch_size,
                                device=args.device,
                                early_stop_patience=5,
                                xval=xval,
                                yval=yval)            
            preds_train = trainer.fit(xtrain, ytrain)
            acc_train = accuracy_fn(preds_train, ytrain)
            f1_train = macrof1_fn(preds_train, ytrain)
            print(f"Train: acc = {acc_train:.3f}% - F1 = {f1_train:.6f}")

            if args.test:
                preds = trainer.predict(xtest)
                acc = accuracy_fn(preds, y_test)
                f1 = macrof1_fn(preds, y_test)
                print(f"Test: acc = {acc:.3f}% - F1 = {f1:.6f}")
            else:
                preds = trainer.predict(xval)
                acc = accuracy_fn(preds, yval)
                f1 = macrof1_fn(preds, yval)
                print(f"Validation: acc = {acc:.3f}% - F1 = {f1:.6f}")

            results.append((lr, max_iters, acc, f1))

    # Plot the results (heatmap)
    learning_rates = sorted(list(set([r[0] for r in results])))
    max_iters_list = sorted(list(set([r[1] for r in results])))
    acc_matrix = np.zeros((len(learning_rates), len(max_iters_list)))
    f1_matrix = np.zeros((len(learning_rates), len(max_iters_list)))

    for (lr, iters, acc, f1) in results:
        i = learning_rates.index(lr)
        j = max_iters_list.index(iters)
        acc_matrix[i, j] = acc
        f1_matrix[i, j] = f1

    fig, ax = plt.subplots()
    im = ax.imshow(f1_matrix, cmap='viridis')

    ax.set_xticks(np.arange(len(max_iters_list)))
    ax.set_yticks(np.arange(len(learning_rates)))
    ax.set_xticklabels(max_iters_list)
    ax.set_yticklabels([f"{lr:.0e}" for lr in learning_rates])

    plt.xlabel("max_iters")
    plt.ylabel("learning rate")
    plt.title("Validation/Test F1-score")

    # Annotate values on heatmap
    for i in range(len(learning_rates)):
        for j in range(len(max_iters_list)):
            value = f"{f1_matrix[i, j]:.2f}"
            ax.text(j, i, value, ha="center", va="center", color="white" if f1_matrix[i, j] < 0.5 else "black")

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    parser.add_argument('--early_stop_patience', type=int, default=5,
                    help="Number of epochs to wait for improvement before early stopping")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
