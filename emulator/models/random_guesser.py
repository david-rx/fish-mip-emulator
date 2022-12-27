import numpy as np

from emulator.models.model import Model

class RandomGuesser(Model):

    def __init__(self) -> None:
        self.train_mean = 0
        self.train_std = 0
        label_dim = 1

    def fit(self, train_examples, train_labels):
        self.train_mean = train_labels.mean()
        self.train_std = train_labels.std()
        self.label_dim = train_labels.shape[1]


    def predict(self, data):
        return np.random.normal(self.train_mean, self.train_std, (data.shape[0], self.label_dim))