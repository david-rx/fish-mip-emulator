from abc import ABC
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
import numpy as np


class Dataloader(ABC):
    def __init__(self) -> None:
        pass

    def load_train_eval(self) -> Tuple:
        """
        Gets the train and eval splits for inputs and outputs
        """
        tos_train, tos_eval, tcb_train, tcb_eval = train_test_split(self.features, self.labels, test_size=0.1)
        return tos_train, tos_eval, tcb_train, tcb_eval

def filter(input_array: np.ndarray, label_array: np.ndarray, fill_values: List[float], label_fill_value: Optional[float] = None):
    """
    Filter array elements if their rows contain filter values.
    """
    filter_list: List[bool] = []
    for row, label in zip(input_array, label_array):
        to_filter = any([val == fill for val, fill in zip(fill_values, row)])
        if label_fill_value is not None:
            to_filter = to_filter or label == label_fill_value
        filter_list.append(to_filter)
    filter_array = np.ndarray(filter_list)
    return input_array[filter_array], label_array[filter_array]