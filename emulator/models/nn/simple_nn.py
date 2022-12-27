
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 256

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
        # torch.nn.init.constant_(self.final.bias, 30) #Start with the mean roughly!


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # x = self.semifinal_sequence(x)

        x = self.final(x)
        return x

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
