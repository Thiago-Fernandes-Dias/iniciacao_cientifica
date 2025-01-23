from cmu_dataset import *
from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score, recall_score
from two_class_results import TwoClassResults

class TwoClassExperiment:
    cmu_database: CMUDataset
    estimator_factory: Callable[[], BaseEstimator]

    def __init__(self, dataset: CMUDataset, estimator_factory: Callable[[], BaseEstimator]) -> None:
        self.cmu_database = dataset
        self.estimator_factory = estimator_factory
    
    def exec(self) -> TwoClassResults:
        X_training: dict[str, pd.DataFrame] = {}
        y_training: dict[str, list[int]] = {}
        X_user_test: dict[str, pd.DataFrame] = {}
        y_user_test: dict[str, list[int]] = {}
        X_other_test: dict[str, pd.DataFrame] = {}
        y_other_test: dict[str, list[int]] = {}
        estimators_map: dict[str, BaseEstimator] = {}
        estimators_hp_map: dict[str, dict[str, object]] = {}

        for uk in self.cmu_database.user_keys():
            X_training[uk], y_training[uk] = \
                self.cmu_database.one_vs_rest_training_rows(uk)
            X_user_test[uk], y_user_test[uk], X_other_test[uk], y_other_test[uk] = \
                self.cmu_database.one_vs_rest_test_rows(uk)
        
        for uk in self.cmu_database.user_keys():
            estimators_map[uk] = self.estimator_factory().fit(X_training[uk], y_training[uk])
            estimators_hp_map[uk] = estimators_map[uk].get_params(deep=False)
        
        two_class_bacc_map: dict[str, float] = {}
        two_class_recall_map: dict[str, float] = {}
        predictions_on_genuine_samples: dict[str, list[int]] = {}
        predictions_on_impostor_samples: dict[str, list[int]] = {}
        
        for uk in self.cmu_database.user_keys():
            predictions_on_genuine_samples[uk] = estimators_map[uk].predict(X_user_test[uk]).flatten().tolist()
            predictions_on_impostor_samples[uk] = estimators_map[uk].predict(X_other_test[uk]).flatten().tolist()
            predictions = predictions_on_genuine_samples[uk] + predictions_on_impostor_samples[uk]
            y_test = y_user_test[uk] + y_other_test[uk]
            two_class_bacc_map[uk] = balanced_accuracy_score(y_test, predictions)
            two_class_recall_map[uk] = recall_score(y_test, predictions, average='micro')
        
        results = TwoClassResults(two_class_bacc_map, two_class_recall_map, 
                                  predictions_on_genuine_samples, predictions_on_impostor_samples,
                                  estimators_hp_map)

        return results
