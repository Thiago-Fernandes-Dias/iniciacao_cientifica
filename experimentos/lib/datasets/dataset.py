from abc import abstractmethod
from lib.constants import *
from lib.utils import *

import pandas as pd

class Dataset:
    _training_df: pd.DataFrame
    _test_df: pd.DataFrame
    _user_keys: set[str]
    _columns_filter_rg: str
    _seed: int = 0
    _seed_change_cbs: list[Callable[[], None]] = []

    @abstractmethod
    def get_drop_columns(self) -> list[str]:
        pass
    
    @abstractmethod
    def user_key_name(self) -> str:
        pass

    @abstractmethod
    def get_session_key_name(self) -> str:
        pass

    @abstractmethod
    def get_repetition_key_name(self) -> str:
        pass

    def __init__(self, file_path: str, test_train_split: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
                 columns_filer_rg: str = '.*') -> None:
        dataset: pd.DataFrame = pd.read_csv(file_path)
        self._user_keys: set[str] = set(dataset[self.user_key_name()].drop_duplicates().tolist())
        self._training_df, self._test_df = test_train_split(dataset)
        self._columns_filter_rg = columns_filer_rg

    def add_seed_change_cb(self, cb: Callable[[], None]):
        self._seed_change_cbs.append(cb)

    def training_df_query(self, query: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        return query(self._training_df).filter(regex=self._columns_filter_rg)

    def test_df_query(self, query: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        return query(self._test_df).filter(regex=self._columns_filter_rg)

    def user_keys(self) -> set[str]:
        return self._user_keys

    def one_class_training_set(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        vectors = self.user_training_samples(user_subject)
        labels = create_labels(vectors, GENUINE_LABEL)
        return vectors, labels

    def two_class_training_set(self, user_subject: str) -> tuple[pd.DataFrame, list[int], pd.DataFrame, list[int]]:
        genuine_vectors = self.user_training_samples(user_subject)
        impostor_vectors = self \
            .training_df_query(lambda df: df[df[self.user_key_name()] != user_subject]) \
            .sample(n=data_frame_length(genuine_vectors), random_state=self._seed)
        return (genuine_vectors, create_labels(genuine_vectors, GENUINE_LABEL),
                impostor_vectors, create_labels(impostor_vectors, IMPOSTOR_LABEL))

    def impostors_test_set(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        attack_vectors = self.test_df_query(lambda df: df[df[self.user_key_name()] != user_subject])
        labels = create_labels(attack_vectors, IMPOSTOR_LABEL)
        return attack_vectors, labels

    def user_test_set(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        user_vectors = self.test_df_query(lambda df: df[df[self.user_key_name()] == user_subject])
        genuine_labels = create_labels(user_vectors, GENUINE_LABEL)
        return user_vectors, genuine_labels

    def user_training_samples(self, user_subject: str) -> pd.DataFrame:
        return self.training_df_query(lambda df: df[df[self.user_key_name()] == user_subject])

    def multi_class_training_samples(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._training_df, self._training_df[self.user_key_name()]

    def multi_class_test_samples(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._test_df, self._test_df[self.user_key_name()]
    
    def set_seed(self, seed: int):
        self._seed = seed
        for cb in self._seed_change_cbs:
            cb()
