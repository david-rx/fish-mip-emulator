from abc import ABC
import time
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

def filter_fill(input_array: np.ndarray, label_array: np.ndarray, fill_values: Iterable[float], label_fill_value: Optional[float]):
    """
    Exclude rows containing filter values.

    Input array: np.ndarray (N, 3)
    """
    assert len(fill_values) == input_array.shape[1]

    anyfill = input_array == fill_values[0]
    for fill_value in fill_values:
        anyfill = anyfill | (input_array == fill_value)

    filter_array = ~np.any(anyfill, axis = 1)
    # filter_array = filter_array_old & (label_array != label_fill_value)

    # assert np.all(filter_array_old == filter_array)

    return input_array[filter_array], label_array[filter_array]


def __filter_fill_dep(input_array: np.ndarray, label_array: np.ndarray, fill_values: Iterable[float], label_fill_value: Optional[float] = None):
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

def test_new_filter():
    array = np.zeros((20000000, 3))
    label_array = np.zeros(20000000)
    fill_values = 4, 5, 12
    array[5, 0] = 4
    array[14, 1] = 8
    array[44, 1] = 5
    array[49, 2] = 12
    start_time = time.time()
    filtered_array, filtered_labels = filter_fill_new(array, label_array, fill_values=fill_values, label_fill_value=32)
    first_filter_finished = time.time()
    new_filter_time = first_filter_finished - start_time
    filtered_array_old, filtered_labels_old = filter_fill(array, label_array, fill_values=fill_values, label_fill_value=None)
    old_filter_time = time.time() - first_filter_finished
    print(f"old time was {old_filter_time} and new time was {new_filter_time}")
    # assert np.all(filtered_array_old == filtered_array)
    # assert np.all(filtered_labels_old == filtered_labels)
    print(filtered_array[0:7])
    print(filtered_array.shape)
    print(filtered_array_old[0:7])
    print(filtered_array_old.shape)


if __name__ == "__main__":
    test_new_filter()