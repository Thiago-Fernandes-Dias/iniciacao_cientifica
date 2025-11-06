from typing import Callable

import pandas as pd

from lib.datasets.dataset import Dataset


class CMUDataset(Dataset):
    def __init__(self, file_path: str, test_train_split: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
                 columns_filer_rg: str = '.*') -> None:
        super().__init__(file_path, test_train_split, columns_filer_rg)
    
    def get_drop_columns(self) -> list[str]:
        return ['sessionIndex', 'rep', 'subject']
    
    def user_key_name(self) -> str:
        return 'subject'

    def get_session_key_name(self) -> str:
        return 'sessionIndex'
    
    def get_repetition_key_name(self) -> str:
        return 'rep'