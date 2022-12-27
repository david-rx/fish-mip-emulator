BATCH_SIZE = 16
NUM_EPOCHS = 16
LEARNING_RATE = 0.0001

from dataclasses import dataclass
import wandb

from emulator.models.nn.helpers import compute_grad_norm, rebalance
USE_WNB = False

if USE_WNB:
    wandb.init("simple-nn")

from tqdm import tqdm
from emulator.models.model import Model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np

from emulator.models.nn.simple_nn import SimpleNN

"""
Wraps a pytorch neural network in a SKLEARN compatible model.
Supports training, evaluation, and prediction.
"""


@dataclass
class NNConfig:
    """
    NN Config class:
    specify input size, output size, and training
    hyperparameters here.
    """
    input_size: int
    output_size: int
    lr: float = LEARNING_RATE
    batch_size: int = BATCH_SIZE
    num_epochs: int = NUM_EPOCHS
    lr_scheduler: str = "exponential"
    optimizer: str = "adamw"

 
class TensorTupleDataset(Dataset):

    def __init__(self, first_tensor: torch.Tensor, second_tensor: torch.Tensor) -> None:
        self.first_tensor = first_tensor
        self.second_tensor = second_tensor

    def __len__(self):
        return len(self.first_tensor)

    def __getitem__(self, index: int) -> torch.tensor:
        return self.first_tensor[index], self.second_tensor[index]

class NNRegressor(Model):
    """
    Makes a pytorch neural network compatible with the SKLEARN api.
    """

    def __init__(self, input_size: int, output_size: int, lr=LEARNING_RATE) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.net = SimpleNN(input_size = input_size, output_size = output_size).to(self.device)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=.9)

    def train(self, train_data: torch.tensor, train_labels: torch.tensor, evaluation_data: torch.Tensor = None, evaluation_labels: torch.Tensor = None):

        dataset = TensorTupleDataset(first_tensor = train_data, second_tensor=train_labels)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)
        self.net.train()
        log_steps = 50

        for epoch in tqdm(range(NUM_EPOCHS)):

            for index, batch in enumerate(dataloader):


                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                predictions = self.net.forward(inputs)
                loss = self.loss_fn(predictions, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10000)

                if index % log_steps == 0:
                    grad_norm = compute_grad_norm(self.net)
                    if USE_WNB:
                        wandb.log({"train loss": loss, "epoch": epoch, "grad norm": grad_norm, "predicted max": predictions.max(), "labels max": labels.max()})

                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()

            if evaluation_data is not None:
                eval_score = self.evaluate(evaluation_data, evaluation_labels)
                if USE_WNB:
                    wandb.log({"eval result": eval_score})

        self.net.eval()

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray, evaluation_data: torch.Tensor = None, evaluation_labels: torch.Tensor = None, rebalance_data = False) -> None:
        print(f"starting fit")
        train_data = train_data.astype(np.float32)
        train_labels = train_labels.astype(np.float32)
        train_data_tensor = torch.tensor(train_data)
        train_labels_tensor = torch.tensor(train_labels)
        self.train(train_data=train_data_tensor, train_labels=train_labels_tensor, evaluation_data=evaluation_data, evaluation_labels=evaluation_labels)

    def predict(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        data_tensor = torch.tensor(data)
        dataset = TensorTupleDataset(first_tensor = data_tensor, second_tensor=data_tensor)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE)

        all_predictions = []
        with torch.no_grad():
            for batch in dataloader:
                predictions = self.net(batch[0].to(self.device))
                all_predictions.extend(predictions.cpu().numpy())

        return np.concatenate(all_predictions)

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(data)
        print(f"predicted mean: {predictions.mean()} and labels mean: {labels.mean()}")
        print(f"predicted max: {predictions.max()} and labels max {labels.max()}")
        errors = (predictions - labels.flatten()) ** 2

        return errors.mean()