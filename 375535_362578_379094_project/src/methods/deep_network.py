import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import numpy as np

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
        #[512,256,128,64]
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
        self.fc4 = nn.Linear(hidden_layer_size[2],hidden_layer_size[3]) #so 4 hidden layers? is better
        self.bn4 = nn.BatchNorm1d(hidden_layer_size[3])
        self.fc5 = nn.Linear(hidden_layer_size[3],n_classes)
        #self.bn5 = nn.BatchNorm1d(hidden_layer_size[4])
        #self.fc6 = nn.Linear(hidden_layer_size[4],n_classes)

        self.dropout = nn.Dropout(p=0.3)


        # ed #321: we can add dropout to prevent overfitting
        # ed #328: we can use BatchNorm between the layers //Applies Batch Normalization over a 2D or 3D input.

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        #We want to flatten the image ?
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        #x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        #x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        #x = self.dropout3(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        #x = self.dropout4(x)
        #x = F.relu(self.bn5(self.fc5(x)))
        #x = self.dropout(x)
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
            nn.Flatten(),  # (N, 128 * 7 * 7)
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

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = None
        self.optimizer = None

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

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

        ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()
        for x_batch, y_batch in dataloader:
            self.optimizer.zero_grad()
            preds = self.model(x_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            self.optimizer.step()

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
                x_batch = batch[0]  # (input only)
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
        training_data = torch.from_numpy(training_data).float()
        training_labels = torch.from_numpy(training_labels).long()

        # Compute class weights using training labels
        class_weights = self.compute_class_weights(training_labels)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer setup after model is on correct device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Build DataLoader
        train_dataset = TensorDataset(training_data, training_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data.numpy())

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
