import json
import os
import re
from datetime import datetime
from typing import Callable, TypeVar

import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score

T = TypeVar('T')

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


def item_with_max_value(map: dict[T, float], comp: Callable[[float, float], int]) -> tuple[T, float]:
    max_item, max_value = map.popitem()
    for (item, value) in map.items():
        if comp(value, max_value) == 1:
            max_value = value
            max_item = item
    return (max_item, max_value)


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


def save_results(name: str, experiment_results: dict[str, object]) -> None:
    json_string = json.dumps(experiment_results, indent=4)
    create_dir_if_not_exists("results")
    with open(f"results/{name}_{get_datetime()}.json", 'w+') as file:
        file.write(json_string)


def first_session_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[df['sessionIndex'] == 1], df[df['sessionIndex'] != 1]


def lw_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df[(df['sessionIndex'] == 1) & (df['rep'] <= 12)], df[df['sessionIndex'] != 1]


far_score = make_scorer(lambda y_true, y_pred: 1 - accuracy_score(y_true, y_pred), greater_is_better=False)
