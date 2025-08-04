from abc import abstractmethod, ABC
from datetime import datetime
import logging

import pandas as pd

from lib.datasets.dataset import Dataset
from lib.repositories.results_repository import ResultsRepository
from lib.user_model_prediction import UserModelPrediction
from lib.utils import log_completion


class ExperimentRunner(ABC):
    _dataset: Dataset
    _results_repository: ResultsRepository
    _exp_name: str
    _use_impostor_samples: bool
    _X_genuine_training: dict[str, pd.DataFrame]
    _y_genuine_training: dict[str, list[int]]
    _X_impostor_training: dict[str, pd.DataFrame]
    _y_impostor_training: dict[str, list[int]]
    _X_genuine_test: dict[str, pd.DataFrame]
    _y_genuine_test: dict[str, list[int]]
    _X_impostors_test: dict[str, pd.DataFrame]
    _y_impostors_test: dict[str, list[int]]
    _one_class_estimators_hp_map: dict[str, list[dict[str, object]]]

    def __init__(self, dataset: Dataset, results_repo: ResultsRepository, exp_name: str,
                 use_impostor_samples: bool = False):
        self._dataset = dataset
        self._results_repository = results_repo
        self._exp_name = exp_name
        self._use_impostor_samples = use_impostor_samples
        self._X_genuine_training = {}
        self._y_genuine_training = {}
        self._X_impostor_training = {}
        self._y_impostor_training = {}
        self._X_genuine_test = {}
        self._y_genuine_test = {}
        self._X_impostors_test = {}
        self._y_impostors_test = {}
        self._one_class_estimators_hp_map = {}

        self._set_vectors_and_true_labels()
        self._dataset.add_seed_change_cb(self._set_vectors_and_true_labels)

        self.logger = logging.getLogger(__name__)

    def add_name_suffix(self, s: str):
        self._exp_name += f" ({s})"

    @abstractmethod
    def exec(self) -> None:
        pass

    def _set_vectors_and_true_labels(self, seed: int = 0) -> None:
        for uk in self._dataset.user_keys():
            self._X_genuine_training[uk], self._y_genuine_training[uk], \
                self._X_impostor_training[uk], self._y_impostor_training[uk] = self._dataset.two_class_training_set(uk)
            self._X_genuine_test[uk], self._y_genuine_test[uk] = self._dataset.user_test_set(uk)
            self._X_impostors_test[uk], self._y_impostors_test[uk] = self._dataset.impostors_test_set(uk)

    def _get_user_training_vectors(self, uk):
        x_training, y_training = self._X_genuine_training[uk], self._y_genuine_training[uk]
        if self._use_impostor_samples:
            x_training = pd.concat([x_training, self._X_impostor_training[uk]])
            y_training = y_training + self._y_impostor_training[uk]
        return x_training, y_training

    def _test_user_model(self, estimator, uk, seed: int = 0) -> list[pd.Series]:
        pred_frames = []
        x_test = pd.concat([self._X_genuine_test[uk], self._X_impostors_test[uk]])
        y_test = self._y_genuine_test[uk] + self._y_impostors_test[uk]
        for (_, x), y in zip(x_test.iterrows(), y_test):
            x_filtered = pd.DataFrame([x.drop(labels=self._dataset.get_drop_columns())])
            pred = UserModelPrediction(
                user_id=uk,
                expected=y,
                predicted=estimator.predict(x_filtered)[0].item(),
                session=x[self._dataset.get_session_key_name()],
                repetition=x[self._dataset.get_repetition_key_name()],
            )
            pred_dict = pred.to_dict()
            pred_dict['seed'] = seed
            pred_frame = pd.Series(pred_dict)
            pred_frames.append(pred_frame)
        return pred_frames

    def _log_experiment_completion(self, start_time):
        log_completion(logger=self.logger, msg=f"Experiment {self._exp_name} finished", start_time=start_time)
