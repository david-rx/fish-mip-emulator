import numpy as np

from emulator.models.model import Model

class RandomGuesser(Model):

    def __init__(self) -> None:
        self.train_mean = 0
        self.train_std = 0

    def fit(self, train_examples, train_labels):
        self.train_mean = train_labels.mean()
        self.train_std = train_labels.std()

    def predict(self, data):
        return np.random.normal(self.train_mean, self.train_std, (data.shape[0], 1))