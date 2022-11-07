import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.utils.data import Dataset
import torch.optim as optim

import numpy as np

WIDTH = 128
BATCH_SIZE = 2048

class SimpleNN(torch.module):

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(2, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.final = nn.Linear(WIDTH, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final(x)

class NNRegressor:
    """
    Makes a pytorch neural network compatible with the SKLEARN api.
    """

    def __init__(self, ) -> None:
        self.device = "cuda"
        self.net = SimpleNN().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()


    def train(self, train_data: torch.tensor, train_labels: torch.tensor):

        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)
        dataset = Dataset(train_data, BATCH_SIZE)

        for batch in dataset:
            predictions = self.forward(batch[0])
            loss = self.loss_fn(predictions, batch[1])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray) -> None:
        train_data_tensor = torch.tensor(train_data)
        train_labels_tensor = torch.tensor(train_labels)
        self.train(train_data=train_data_tensor, train_labels=train_labels_tensor)

    def predict(self, data: np.ndarray) -> np.ndarray:
        data_tensor = torch.tensor(data)
        dataset = Dataset(data_tensor, BATCH_SIZE)

        all_predictions = []
        for batch in dataset:
            predictions = self.net(batch)
            all_predictions.extend(predictions.cpu().numpy())

        return all_predictions
