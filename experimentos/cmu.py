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

    def one_vs_one_training_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        vectors = self.training_df_query(lambda df: df[df['subject'] == user_subject])
        labels = create_labels(vectors, GENUINE_LABEL)
        return (vectors, labels)
    
    def one_vs_one_test_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        vectors = self.test_df_query(lambda df: df[df['subject'] == user_subject])
        labels = create_labels(vectors, GENUINE_LABEL)
        return (vectors, labels)

    def one_vs_rest_training_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        user_vectors = self.training_df_query(lambda df: df[df['subject'] == user_subject])
        other_vectors = self\
            .training_df_query(lambda df: df[df['subject'] != user_subject])\
            .sample(n=data_frame_length(user_vectors), random_state=RANDOM_STATE)
        return (pd.concat([user_vectors, other_vectors]), 
                create_labels(user_vectors, GENUINE_LABEL) + create_labels(other_vectors, IMPOSTOR_LABEL))

    def one_vs_rest_test_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int], pd.DataFrame, list[int]]:
        user_vectors = self.test_df_query(lambda df: df[df['subject'] == user_subject])
        other_vectors = self.test_df_query(lambda df: df[df['subject'] != user_subject])
        return (user_vectors, create_labels(user_vectors, GENUINE_LABEL), 
                other_vectors, create_labels(other_vectors, IMPOSTOR_LABEL))

    def one_vs_one_attacks_rows(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        attack_vectors = self.test_df_query(lambda df: df[df['subject'] != user_subject]) 
        labels = create_labels(attack_vectors, IMPOSTOR_LABEL)
        return (attack_vectors, labels)

    