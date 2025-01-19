import pandas as pd

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
    max_value = 0
    max_item = None
    for (item, value) in map.items():
        if value > max_value:
            max_value = value
            max_item = item
    return (max_item, max_value)