import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import numpy as np

from src.utils import accuracy_fn, macrof1_fn

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_layer_size=None):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        if hidden_layer_size is None:
            hidden_layer_size = [input_size // 2,
                input_size // 4,
                input_size // 8,
                input_size // 16,
                input_size // 32
            ]
        
        self.fc1 = nn.Linear(input_size, hidden_layer_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_layer_size[0])
        self.fc2 = nn.Linear(hidden_layer_size[0], hidden_layer_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_layer_size[1])
        self.fc3 = nn.Linear(hidden_layer_size[1], hidden_layer_size[2])
        self.bn3 = nn.BatchNorm1d(hidden_layer_size[2])
        self.fc4 = nn.Linear(hidden_layer_size[2],hidden_layer_size[3])
        self.bn4 = nn.BatchNorm1d(hidden_layer_size[3])
        self.fc5 = nn.Linear(hidden_layer_size[3],n_classes)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        preds = self.fc5(x)
        return preds
        


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.3)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )


    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.conv_layers(x)
        preds = self.fc_layers(x)
        
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, device="cpu", early_stop_patience=10, xval=None, yval=None):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.loss_history = []
        self.val_loss_history = []
        self.val_f1_history = []

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model = model.to(self.device)
        print(f"Using device: {self.device}")

        self.criterion = None
        self.optimizer = None

        self.early_stop_patience = early_stop_patience
        self.xval = xval
        self.yval = yval
        self.best_f1 = 0
        self.epochs_without_improvement = 0

        self.generator = torch.Generator()

    def compute_class_weights(self, labels_tensor):
        """
        Compute class weights inversely proportional to class frequencies.
        Arguments:
            labels (numpy array): training labels
        Returns:
            weights (Tensor): shape (C,) for CrossEntropyLoss
        """
        labels = labels_tensor.cpu().numpy()
        counts = Counter(labels.tolist())
        total = sum(counts.values())
        num_classes = int(labels_tensor.max().item()) + 1
        weights = [np.log(total / (counts.get(i, 1) + 1)) for i in range(num_classes)]
        weights = torch.tensor(weights, dtype=torch.float32)
        return weights
    
    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0)
            
        avg_loss = total_loss / len(dataloader.dataset)
        self.loss_history.append(avg_loss)

    def train_all(self, dataloader):
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

            # Evaluate on validation set if available
            if self.xval is not None and self.yval is not None:
                self.model.eval()
                with torch.no_grad():
                    x_val_tensor = torch.from_numpy(self.xval).float().to(self.device)
                    y_val_tensor = torch.from_numpy(self.yval).long().to(self.device)
                    val_preds_logits = self.model(x_val_tensor)
                    val_loss = self.criterion(val_preds_logits, y_val_tensor).item()
                    val_preds = val_preds_logits.argmax(dim=1).cpu().numpy()
                    val_f1 = macrof1_fn(val_preds, self.yval)

                    self.val_loss_history.append(val_loss)
                    self.val_f1_history.append(val_f1)

                print(f" | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}", end="")

                # early stopping logic
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.early_stop_patience:
                        print("\nEarly stopping triggered.")
                        break

            # Print progress bar
            progress = int((ep + 1) / self.epochs * 50)
            bar = "[" + "■" * progress + " " * (50 - progress) + "]"
            print(f"\rEpoch {ep+1}/{self.epochs} {bar}", end="")

        print()

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_labels = []

        with torch.no_grad():
            for batch in dataloader:
                x_batch = batch[0].to(self.device)
                preds = self.model(x_batch)
                pred_classes = preds.argmax(dim=1)
                pred_labels.append(pred_classes)

        return torch.cat(pred_labels, dim=0)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        training_data = torch.from_numpy(training_data).float().to(self.device)
        training_labels = torch.from_numpy(training_labels).long().to(self.device)

        # Compute class weights using training labels
        class_weights = self.compute_class_weights(training_labels).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer setup after model is on correct device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Build DataLoader
        train_dataset = TensorDataset(training_data, training_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, generator=self.generator)

        self.train_all(train_dataloader)

        return self.predict(training_data.detach().cpu().numpy())

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
