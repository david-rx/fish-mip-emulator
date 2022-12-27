import numpy as np

from emulator.models.model import Model

class MeanGuesser(Model):

    def __init__(self) -> None:
        self.train_mean = 0
        self.label_dim = 1



    def fit(self, train_examples, train_labels):
        self.train_mean = train_labels.mean()
        self.label_dim = train_labels.shape[1]

    def predict(self, data):
        return np.zeros((data.shape[0], self.label_dim)) + self.train_mean