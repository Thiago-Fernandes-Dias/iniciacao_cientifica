import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from lib.cmu_dataset import CMUDataset
from lib.constants import RANDOM_STATE, N_JOBS
from lib.hp_grids import rf_params_grid
from lib.runners.one_class_experiment_with_search_cv_runner_impl import (
    OneClassExperimentWithSearchCVRunnerImpl,
)
from lib.utils import first_session_split, save_results


def main() -> None:
    cmu_database = CMUDataset(
        "datasets/cmu/DSL-StrongPasswordData.csv", first_session_split
    )
    two_class_rf_grid_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    two_class_rf_gs = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=rf_params_grid,
        cv=two_class_rf_grid_cv,
        n_jobs=N_JOBS,
        scoring="accuracy",
    )
    two_class_rf_with_hpo_experiment = OneClassExperimentWithSearchCVRunnerImpl(
        dataset=cmu_database, estimator=two_class_rf_gs, use_impostor_samples=True
    )
    results = two_class_rf_with_hpo_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())


if __name__ == "__main__":
    main()
