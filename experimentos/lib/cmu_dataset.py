from lib.constants import *
from lib.utils import *


class CMUDataset:
    _training_df: pd.DataFrame
    _test_df: pd.DataFrame
    _user_keys: set[str]
    _drop_columns = ['subject', 'sessionIndex', 'rep']
    _columns_filter_rg: str

    def __init__(self, file_path: str, test_train_split: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
                 columns_filer_rg: str = '.*') -> None:
        cmu: pd.DataFrame = pd.read_csv(file_path)
        self._user_keys: set[str] = set(cmu["subject"].drop_duplicates().tolist())
        self._training_df, self._test_df = test_train_split(cmu)
        self._columns_filter_rg = columns_filer_rg

    def training_df_query(self, query: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        return query(self._training_df).drop(columns=self._drop_columns).filter(regex=self._columns_filter_rg)

    def test_df_query(self, query: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        return query(self._test_df).drop(columns=self._drop_columns).filter(regex=self._columns_filter_rg)

    def user_keys(self) -> set[str]:
        return self._user_keys

    def one_class_training_set(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        vectors = self.user_training_samples(user_subject)
        labels = create_labels(vectors, GENUINE_LABEL)
        return vectors, labels

    def two_class_training_set(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        user_vectors = self.user_training_samples(user_subject)
        other_vectors = self \
            .training_df_query(lambda df: df[df['subject'] != user_subject]) \
            .sample(n=data_frame_length(user_vectors), random_state=RANDOM_STATE)
        return (pd.concat([user_vectors, other_vectors]),
                create_labels(user_vectors, GENUINE_LABEL) + create_labels(other_vectors, IMPOSTOR_LABEL))

    def impostors_test_set(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        attack_vectors = self.test_df_query(lambda df: df[df['subject'] != user_subject])
        labels = create_labels(attack_vectors, IMPOSTOR_LABEL)
        return attack_vectors, labels

    def user_test_set(self, user_subject: str) -> tuple[pd.DataFrame, list[int]]:
        user_vectors = self.test_df_query(lambda df: df[df['subject'] == user_subject])
        genuine_labels = create_labels(user_vectors, GENUINE_LABEL)
        return user_vectors, genuine_labels

    def user_training_samples(self, user_subject: str) -> pd.DataFrame:
        return self.training_df_query(lambda df: df[df['subject'] == user_subject])

    def multi_class_training_samples(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._training_df.drop(columns=self._drop_columns), self._training_df['subject']

    def multi_class_test_samples(self) -> tuple[pd.DataFrame, pd.Series]:
        return self._test_df.drop(columns=self._drop_columns), self._test_df['subject']
