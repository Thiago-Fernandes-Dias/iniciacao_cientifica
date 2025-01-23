import pandas as pd
import os

from datetime import datetime
from typing import TypeVar

T = TypeVar('T')

def n_copies(element: T, n: int) -> list[T]:
    return [element] * n

def data_frame_length(df: pd.DataFrame) -> int:
    return df.shape[0]

def create_labels(df: pd.DataFrame, label: T) -> list[T]:
    return n_copies(label, data_frame_length(df))

def float_range(start: float, end: float, step: float) -> list[float]:
    result: list[float] = []
    while start <= end:
        result.append(start)
        start += step
    return result

def item_with_max_value(map: dict[T, float]) -> tuple[T, float]:
    max_item, max_value = map.popitem()
    for (item, value) in map.items():
        if value > max_value:
            max_value = value
            max_item = item
    return (max_item, max_value)

def dict_values_average(map: dict[T, float]) -> float:
    sum = 0.0
    length = 0
    for _, value in map.items():
        length += 1
        sum += value
    return sum / length

def get_datetime() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def create_dir_if_not_exists(name: str):
    if not os.path.exists('results'):
        os.makedirs('results')