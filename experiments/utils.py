import pandas as pd

from typing import TypeVar

T = TypeVar('T')

def n_copies(element: T, n: int) -> list[T]:
    return [element] * n

def data_frame_length(df: pd.DataFrame) -> int:
    return df.shape[0]

def create_labels(df: pd.DataFrame, label: T) -> list[T]:
    return n_copies(label, data_frame_length(df))