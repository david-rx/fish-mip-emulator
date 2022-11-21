from emulator.models.model import Model
import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.utils.data import Dataset
import torch.optim as optim

import numpy as np

WIDTH = 128
BATCH_SIZE = 4096

class TensorTupleDataset(Dataset):

    def __init__(self, first_tensor: torch.Tensor, second_tensor: torch.Tensor) -> None:
        self.first_tensor = first_tensor
        self.second_tensor = second_tensor

    def __len__(self):
        return len(self.first_tensor)

    def __getitem__(self, index: int) -> torch.tensor:
        return self.first_tensor[index], self.second_tensor[index]


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

class NNRegressor(Model):
    """
    Makes a pytorch neural network compatible with the SKLEARN api.
    """

    def __init__(self, ) -> None:
        self.device = "cuda"
        self.net = SimpleNN().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()


    def train(self, train_data: torch.tensor, train_labels: torch.tensor):

        dataset = TensorTupleDataset(first_tensor = train_data, second_tensor=train_labels)
        dataloader = torch.utils.data.Dataloader(dataset, batch_size = BATCH_SIZE)

        for batch in dataloader:
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            predictions = self.forward(inputs)
            loss = self.loss_fn(predictions, labels)
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
