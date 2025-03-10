import os

import numpy as np
from sklearn.model_selection import KFold

from lib.cmu_dataset import CMUDataset
from lib.lightweight_alg import LightWeightAlg
from lib.one_class_threshold_search_cv import OneClassThresholdSearchCV
from lib.runners.one_class_experiment_with_treshold_search_cv import OneClassExperimentWithTresholdSearchCV
from lib.utils import lw_split, save_results, first_session_split, float_range


def main() -> None:
    cmu_database = CMUDataset("datasets/cmu/DSL-StrongPasswordData.csv", first_session_split)
    one_class_lw_grid_cv = KFold(n_splits=5)
    one_class_lw_gs = OneClassThresholdSearchCV(
        estimator=LightWeightAlg(),
        cv=one_class_lw_grid_cv,
        thresholds=float_range(0.2, 0.7, 0.025),
    )
    one_class_lw_with_hpo_experiment = OneClassExperimentWithTresholdSearchCV(
        dataset=cmu_database, estimator=one_class_lw_gs
    )
    results = one_class_lw_with_hpo_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())


if __name__ == "__main__":
    main()
