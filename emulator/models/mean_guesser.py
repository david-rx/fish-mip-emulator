import numpy as np

class MeanGuesser:

    def __init__(self) -> None:
        self.train_mean = 0


    def fit(self, train_examples, train_labels):
        self.train_mean = train_labels.mean()

    def predict(self, data):
        return np.zeros(data.shape[0]) + self.train_mean