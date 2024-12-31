import pandas as pd

from typing import Callable
from utils import *

GENUINE_LABEL = 1
IMPOSTOR_LABEL = -1
RANDOM_STATE = 42

class CMUDatabase:
    _training_df: pd.DataFrame
    _test_df: pd.DataFrame
    _user_keys: set[str]
    _drop_columns = ['subject', 'sessionIndex', 'rep']
    
    def __init__(self, file_path: str):
        cmu: pd.DataFrame = pd.read_csv(file_path)
        self.training_df = cmu[cmu['sessionIndex'] == 1]
        self.test_df = cmu[cmu['sessionIndex'] != 1]
        self._user_keys: set[str] = set(cmu["subject"].drop_duplicates().tolist())

    def training_df_query(self, query: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        return query(self.training_df).drop(columns=self._drop_columns)
    
    def test_df_query(self, query: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        return query(self.test_df).drop(columns=self._drop_columns)
    
    def user_keys(self) -> set[str]:
        return self._user_keys

    def user_training_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        vectors = self.training_df_query(lambda df: df[df['subject'] == user_subject])
        labels = create_labels(vectors, GENUINE_LABEL)
        return (vectors, labels)
    
    def other_training_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        others_typing_samples = self.training_df_query(lambda df: df[df['subject'] != user_subject])
        vectors = others_typing_samples.sample(n=50, random_state=RANDOM_STATE)
        labels = create_labels(vectors, IMPOSTOR_LABEL)
        return (vectors, labels)

    def user_test_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        vectors = self.test_df_query(lambda df: df[df['subject'] == user_subject])
        labels = create_labels(vectors, GENUINE_LABEL)
        return (vectors, labels)
    
    def other_test_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        vectors = self.test_df_query(lambda df: df[df['subject'] != user_subject])
        labels = create_labels(vectors, IMPOSTOR_LABEL)
        return (vectors, labels)