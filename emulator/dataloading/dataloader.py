from abc import ABC
from typing import Iterable, List, Optional, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


class Dataloader(ABC):
    def __init__(self) -> None:
        pass

    def load_train_eval(self) -> Tuple:
        """
        Gets the train and eval splits for inputs and outputs
        """
        tos_train, tos_eval, tcb_train, tcb_eval = train_test_split(self.features, self.labels, test_size=0.1)
        return tos_train, tos_eval, tcb_train, tcb_eval

def filter_fill(input_array: np.ndarray, label_array: np.ndarray, fill_values: Iterable[float], label_fill_value: Optional[float] = None):
    """
    Filter array elements if their rows contain filter values.
    """
    assert len(fill_values) == input_array.shape[1]
    print("starting filter")

    filter_list: List[bool] = []
    for row, label in tqdm(zip(input_array, label_array)):
        to_keep = not any([val == fill for val, fill in zip(fill_values, row)])
        if label_fill_value is not None:
            to_keep = to_keep and (label[0] != label_fill_value)
        filter_list.append(to_keep)
    filter_array = np.asarray(filter_list)
    print("filtered")
    return input_array[filter_array], label_array[filter_array]

def filter_fill_by_period(input_array: np.ndarray, label_array: np.ndarray, fill_values: Iterable[float], label_fill_value: Optional[float] = None):
    """
    Filter array elements if their rows contain filter values.
    """
    assert len(fill_values) == input_array.shape[-1]
    print("starting filter")
    filtered_arrays = []
    filtered_label_arrays = []
    filter_list = []


    period_row, period_labels = input_array[0], label_array[0]
    for row, label in tqdm(zip(period_row, period_labels)):
        to_keep = not any([val == fill for val, fill in zip(fill_values, row)])
        if label_fill_value is not None:
            to_keep = to_keep and (label[0] != label_fill_value)
        filter_list.append(to_keep)
    filter_array = np.asarray(filter_list)

    for period_row, period_labels in zip(input_array, label_array):
        filtered_arrays.append(period_row[filter_array])
        filtered_label_arrays.append(period_labels[filter_array])

    print("filtered")
    return np.stack(filtered_arrays, axis=0), np.stack(filtered_label_arrays, axis = 0)
