from abc import ABC
import time
from typing import Iterable, List, Optional, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


class Dataloader:
    def __init__(self) -> None:
        pass

    def load_train_eval(self) -> Tuple:
        """
        Gets the train and eval splits for inputs and outputs
        """
        tos_train, tos_eval, tcb_train, tcb_eval = train_test_split(self.features, self.labels, test_size=0.1)
        return tos_train, tos_eval, tcb_train, tcb_eval
    
    def _print_stats(self):
        print(f"features shape: {self.features.shape}")
        print(f"labels shape: {self.labels.shape}")
        print(f"labels max and min: {self.labels.max()} and {self.labels.min()}")
        print(f"features max and min: {self.features.max()} and {self.features.min()}")

def filter_fill(input_array: np.ndarray, label_array: np.ndarray, fill_values: Iterable[float], label_fill_value: Optional[float]):
    """
    Exclude rows containing filter values.

    Input array: np.ndarray (N, 3)
    """
    anyfill = input_array == fill_values[0]
    for fill_value in fill_values:
        anyfill = anyfill | (input_array == fill_value)

    filter_array = ~np.any(anyfill, axis = 1)
    filter_array = filter_array & (label_array.reshape(-1) != label_fill_value)

    return input_array[filter_array], label_array[filter_array]

def filter_fill_by_period(input_array: np.ndarray, label_array: np.ndarray, fill_values: Iterable[float], label_fill_value: Optional[float] = None):
    """
    Rewrite of filter_fill to filter by period.
    Uses the same filter_fill function.
    """
    filtered_arrays = []
    filtered_label_arrays = []

    for period_row, period_labels in zip(input_array, label_array):

        per_filtered_input_arrays, per_filtered_label_arrays = filter_fill(period_row, period_labels, fill_values, label_fill_value)
        filtered_arrays.append(per_filtered_input_arrays)
        filtered_label_arrays.append(per_filtered_label_arrays)
    return np.stack(filtered_arrays, axis=0), np.stack(filtered_label_arrays, axis = 0)

def replace_fill(input_array: np.ndarray, label_array: np.ndarray, fill_values):
    return zero_out(input_array, fill_values), zero_out(label_array, fill_values)
    
def zero_out(arr, values):
    # Create a mask with True for each element equal to any of the values in the list
    mask = np.isin(arr, values)

    # Create a new array with the same shape as the input array
    result = np.zeros_like(arr)

    # Set the elements in the result array to the corresponding elements in the input array where the mask is False
    result[~mask] = arr[~mask]

    return result

