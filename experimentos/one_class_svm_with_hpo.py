import os

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import OneClassSVM

from lib.cmu_dataset import CMUDataset
from lib.hp_grids import one_class_svm_params_grid
from lib.runners.one_class_experiment_with_search_cv_runner_impl import OneClassExperimentWithSearchCVRunnerImpl
from lib.utils import first_session_split, save_results, exclude_hold_times_pt


def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split,
                              exclude_hold_times_pt)
    one_class_svm_grid_cv = KFold(n_splits=5)
    one_class_svm_gs = GridSearchCV(OneClassSVM(), one_class_svm_params_grid,
                                    scoring='accuracy', cv=one_class_svm_grid_cv, n_jobs=-1)
    one_class_svm_with_hpo_experiment = OneClassExperimentWithSearchCVRunnerImpl(dataset=cmu_database,
                                                                                 estimator=one_class_svm_gs)
    results = one_class_svm_with_hpo_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())


if __name__ == "__main__":
    main()
