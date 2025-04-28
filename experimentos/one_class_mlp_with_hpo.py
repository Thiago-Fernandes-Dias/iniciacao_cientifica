import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier

from lib.datasets.dataset import Dataset
from lib.constants import N_JOBS
from lib.repositories.results_repository import results_repository_factory
from lib.runners.one_class_experiment_with_search_cv_runner_impl import (
    OneClassExperimentWithSearchCVRunnerImpl,
)
from lib.utils import cmu_first_session_split,  save_results
from lib.hp_grids import mlp_params_grid


def main() -> None:
    cmu_database = Dataset(
        "datasets/cmu/DSL-StrongPasswordData.csv", cmu_first_session_split
    )
    one_class_mlp_grid_cv = KFold(n_splits=5)
    one_class_mlp_gs = GridSearchCV(
        MLPClassifier(),
        mlp_params_grid,
        scoring="accuracy",
        cv=one_class_mlp_grid_cv,
        n_jobs=N_JOBS,
    )
    one_class_mlp_with_hpo_experiment = OneClassExperimentWithSearchCVRunnerImpl(
        dataset=cmu_database, estimator=one_class_mlp_gs
    )
    results = one_class_mlp_with_hpo_experiment.exec()
    repo = results_repository_factory()
    repo.add_one_class_result(results, os.path.basename(__file__).replace(".py", ""))


if __name__ == "__main__":
    main()
