from ast import With
from cProfile import label
from random import randint
from matplotlib.widgets import Widget
from tqdm import tqdm
from emulator.models.model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import statistics
import numpy as np

import wandb


WIDTH = 256
BATCH_SIZE = 4 ** 7
NUM_EPOCHS = 3
LEARNING_RATE = 0.0001

wandb.init("simple-nn")

class TensorTupleDataset(Dataset):

    def __init__(self, first_tensor: torch.Tensor, second_tensor: torch.Tensor) -> None:
        self.first_tensor = first_tensor
        self.second_tensor = second_tensor

    def __len__(self):
        return len(self.first_tensor)

    def __getitem__(self, index: int) -> torch.tensor:
        return self.first_tensor[index], self.second_tensor[index]


class SimpleNN(nn.Module):

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_size, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)
        self.fc3 = nn.Linear(WIDTH, WIDTH)
        self.fc4 = nn.Linear(WIDTH, WIDTH)
        self.fc5 = nn.Linear(WIDTH, WIDTH)

        self.semifinal_sequence = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(),
            nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(),
            # nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(),
            nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH), nn.ReLU())

        self.final = nn.Linear(WIDTH, output_size)
        torch.nn.init.constant_(self.final.bias, 30) #Start with the mean roughly!


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.semifinal_sequence(x)

        x = self.final(x)
        return x

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

class NNRegressor(Model):
    """
    Makes a pytorch neural network compatible with the SKLEARN api.
    """

    def __init__(self, input_size: int, output_size: int, lr=LEARNING_RATE) -> None:
        self.device = "cuda"
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
                    wandb.log({"train loss": loss, "epoch": epoch, "grad norm": grad_norm, "predicted max": predictions.max(), "labels max": labels.max()})

                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()

            if evaluation_data is not None:
                eval_score = self.evaluate(evaluation_data, evaluation_labels)
                wandb.log({"eval result": eval_score})

        self.net.eval()

    def fit(self, train_data: np.ndarray, train_labels: np.ndarray, evaluation_data: torch.Tensor = None, evaluation_labels: torch.Tensor = None, rebalance_data = False) -> None:
        print(f"starting fit")
        if rebalance_data:
            train_data, train_labels = rebalance(train_data, train_labels)
            print("data rebalanced")
        train_data_tensor = torch.tensor(train_data)
        train_labels_tensor = torch.tensor(train_labels)
        self.train(train_data=train_data_tensor, train_labels=train_labels_tensor, evaluation_data=evaluation_data, evaluation_labels=evaluation_labels)

    def predict(self, data: np.ndarray) -> np.ndarray:
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

def compute_grad_norm(model: torch.nn.Module):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def rebalance(train_data, train_labels):
    filtered_train_data = []
    filtered_train_labels = []
    filter_array = train_labels > 75
    np.concatenate(filtered_train_data, filtered_train_data[filter_array])
    for ex, label in zip(train_data, train_labels):
        if label[0] > 75:
            for i in range(1):
                filtered_train_data.append(ex)
                filtered_train_labels.append(label)

        # if label < 75:
        #     num = randint(0, 100)
        #     if num < 97:
        #         continue
        # elif label < 400:
        #     num = randint(0, 100)
        #     if num < 92:
        #         continue
        filtered_train_data.append(ex)
        filtered_train_labels.append(label)


    filtered_train_data = np.stack(filtered_train_data)
    filtered_train_labels = np.stack(filtered_train_labels)
    print(filtered_train_labels.shape)
    print(filtered_train_data.shape)

    print(f"reduced train labels from size {len(train_data)} to size {len(filtered_train_data)}")
    print(f"changed mean from {train_labels.mean()} to {filtered_train_labels.mean()}")
    return filtered_train_data, filtered_train_labels

