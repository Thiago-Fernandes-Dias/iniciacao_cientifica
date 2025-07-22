import os
import re
from typing import Callable, TypeVar

import pandas as pd

T = TypeVar('T')
S = TypeVar('S')

exclude_hold_times_pt: str = "((D|U)D)(.[a-zA-Z]+)+"


def n_copies(element: T, n: int) -> list[T]:
    return [element] * n


def data_frame_length(df: pd.DataFrame) -> int:
    return df.shape[0]


def sorted_nicely(l: list[str]):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def create_labels(df: pd.DataFrame, label: T) -> list[T]:
    return n_copies(label, data_frame_length(df))


def float_range(start: float, end: float, step: float) -> list[float]:
    result: list[float] = []
    while start <= end:
        result.append(start)
        start += step
    return result


def select(iterable: list[T], f: Callable[[T], S]) -> list[S]:
    return [f(x) for x in iterable]


def item_with_max_value(d: dict[T, float], comp: Callable[[float, float], int]) -> tuple[T, float]:
    max_item, max_value = d.popitem()
    for (item, value) in d.items():
        if comp(value, max_value) == 1:
            max_value = value
            max_item = item
    return max_item, max_value


def bigger_comp(a: float, b: float) -> int:
    if a > b:
        return 1
    elif a == b:
        return 0
    return -1


def lower_comp(a: float, b: float) -> int:
    if a > b:
        return -1
    elif a == b:
        return 0
    return 1


def dict_values_average(d: dict[T, float]) -> float:
    summation = 0.0
    length = 0
    for _, value in d.items():
        length += 1
        summation += value
    return summation / length



def create_dir_if_not_exists(name: str):
    if not os.path.exists(name):
        os.makedirs(name)


def cmu_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['sessionIndex'] == 1)], df[(df['sessionIndex'] != 1)]


def cmu_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['sessionIndex'] == 1) & (df['rep'] <= 5)], df[(df['sessionIndex'] != 1) & (df['rep'] <= 5)]


def keyrecs_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['session'] == 1) & (df['repetition'] <= 50)], df[(df['session'] == 2) | (df['repetition'] > 50)]


def keyrecs_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['session'] == 1) & (df['repetition'] <= 5)], df[(df['session'] == 2) & (df['repetition'] <= 5)]


seeds_range = range(0, 3)
