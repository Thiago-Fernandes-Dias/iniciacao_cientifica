import os

import numpy as np
from sklearn.model_selection import KFold

from lib.cmu_dataset import CMUDataset
from lib.constants import RANDOM_STATE
from lib.lightweight_alg import LightWeightAlg
from lib.one_class_threshold_search_cv import OneClassThresholdSearchCV
from lib.runners.one_class_experiment_with_treshold_search_cv import OneClassExperimentWithTresholdSearchCV
from lib.utils import save_results, first_session_split, float_range, two_session_split

def exec_with_dataset(dataset: CMUDataset, file_suffix: str) -> None:
    one_class_lw_grid_cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    one_class_lw_gs = OneClassThresholdSearchCV(
        estimator=LightWeightAlg(),
        cv=one_class_lw_grid_cv,
        thresholds=float_range(0.2, 0.9, 0.05),
    )
    one_class_lw_with_hpo_experiment = OneClassExperimentWithTresholdSearchCV(
        dataset=dataset, estimator=one_class_lw_gs
    )
    results = one_class_lw_with_hpo_experiment.exec()
    save_results(f"{os.path.basename(__file__)}_{file_suffix}", results.to_dict())

def main() -> None:
    cmu1 = CMUDataset("datasets/cmu/DSL-StrongPasswordData.csv", two_session_split)
    exec_with_dataset(cmu1, "two_session_split")
    cmu2 = CMUDataset("datasets/cmu/DSL-StrongPasswordData.csv", first_session_split)
    exec_with_dataset(cmu2, "first_session_split")


if __name__ == "__main__":
    main()
