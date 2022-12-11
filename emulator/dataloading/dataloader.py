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
    

# def filter_fill_by_period(input_array: np.ndarray, label_array: np.ndarray, fill_values: Iterable[float], label_fill_value: Optional[float] = None):
#     """
#     Filter array elements if their rows contain filter values.
#     """
#     print("starting filter")
#     assert len(fill_values) == input_array.shape[-1], "fill_values must have same length as input_array columns"
#     filtered_arrays = []
#     filtered_label_arrays = []
#     filter_list = []

#     period_row, period_labels = input_array[0], label_array[0]
#     for row, label in tqdm(zip(period_row, period_labels)):
#         to_keep = not any([val == fill for val, fill in zip(fill_values, row)])
#         if label_fill_value is not None:
#             to_keep = to_keep and (label[0] != label_fill_value)
#         filter_list.append(to_keep)
#     filter_array = np.asarray(filter_list)
#     print(filter_array)

#     for period_row, period_labels in zip(input_array, label_array):
#         filtered_arrays.append(period_row[filter_array])
#         filtered_label_arrays.append(period_labels[filter_array])

#     print("filtered")
#     return np.stack(filtered_arrays, axis=0), np.stack(filtered_label_arrays, axis = 0)
