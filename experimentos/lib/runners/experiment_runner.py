from abc import abstractmethod, ABC

import pandas as pd

from lib.datasets.dataset import Dataset
from lib.one_class_results import *


class ExperimentRunner(ABC):
    _dataset: Dataset
    _use_impostor_samples: bool
    _X_genuine_training: dict[str, pd.DataFrame]
    _y_genuine_training: dict[str, list[int]]
    _X_impostor_training: dict[str, pd.DataFrame]
    _y_impostor_training: dict[str, list[int]]
    _X_genuine_test: dict[str, pd.DataFrame]
    _y_genuine_test: dict[str, list[int]]
    _X_impostors_test: dict[str, pd.DataFrame]
    _y_impostors_test: dict[str, list[int]]
    _predictions_on_genuine_samples_map: dict[str, list[int]]
    _one_class_estimators_hp_map: dict[str, list[dict[str, object]]]
    _user_model_predictions: pd.DataFrame

    def __init__(self, dataset: Dataset, use_impostor_samples: bool = False):
        self._dataset = dataset
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
        self._user_model_predictions = pd.DataFrame()

    @abstractmethod
    def _calculate_predictions(self) -> None:
        pass

    def exec(self) -> OneClassResults:
        self._reset_from_previous_execution()
        self._set_vectors_and_true_labels()
        self._calculate_predictions()
        results = OneClassResults(
            user_model_predictions=self._user_model_predictions,
            hp=pd.DataFrame.from_dict(self._one_class_estimators_hp_map, orient='index'),
            date=datetime.now())
        return results

    def _set_vectors_and_true_labels(self) -> None:
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
        x_training = x_training.drop(columns=self._dataset.get_drop_columns())
        return x_training, y_training

    def _calculate_user_model_predictions(self, estimator, uk, seed: int | None) -> list[pd.Series]:
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
            if seed is not None:
                pred_dict['seed'] = seed
            pred_frame = pd.Series(pred.to_dict())
            pred_frames.append(pred_frame)
        return pred_frames

    def _reset_from_previous_execution(self):
        self._X_genuine_training.clear()
        self._y_genuine_training.clear()
        self._X_impostor_training.clear()
        self._y_impostor_training.clear()
        self._X_genuine_test.clear()
        self._y_genuine_test.clear()
        self._X_impostors_test.clear()
        self._y_impostors_test.clear()
        self._user_model_predictions = pd.DataFrame()
        self._one_class_estimators_hp_map.clear()
