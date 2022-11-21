from abc import ABC
from typing import Any


class Model(ABC):

    def __init__(self) -> None:
        super().__init__()

    def predict(data) -> float:
        raise NotImplementedError()

    def fit(train_examples, train_labels) -> None:
        raise NotImplementedError()
