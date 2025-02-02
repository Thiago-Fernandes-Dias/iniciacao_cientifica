import pandas as pd
import os
import json

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

def save_results(name: str, experiment_results: dict[str, object]) -> None:
    json_string = json.dumps(experiment_results, indent=4)
    create_dir_if_not_exists("results")
    with open(f"results/{name}_{get_datetime()}.json", 'w+') as file:
        file.write(json_string)

def first_session_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (df[df['sessionIndex'] == 1], df[df['sessionIndex'] != 1])

def lw_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (df[(df['sessionIndex'] == 1) & (df['rep'] <= 12)], df[df['sessionIndex'] != 1])