import os

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier

from lib.dataset import Dataset
from lib.constants import RANDOM_STATE, N_JOBS
from lib.hp_grids import mlp_params_grid
from lib.repositories.results_repository import results_repository_factory
from lib.runners.one_class_experiment_with_search_cv_runner_impl import (
    OneClassExperimentWithSearchCVRunnerImpl,
)
from lib.utils import cmu_first_session_split,  save_results


def main() -> None:
    cmu_database = Dataset(
        "datasets/cmu/DSL-StrongPasswordData.csv", cmu_first_session_split
    )
    one_vs_rest_mlp_grid_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    one_vs_rest_mlp_gs = GridSearchCV(
        estimator=MLPClassifier(),
        param_grid=mlp_params_grid,
        cv=one_vs_rest_mlp_grid_cv,
        n_jobs=N_JOBS,
        scoring="accuracy",
    )
    two_class_mlp_with_hpo = OneClassExperimentWithSearchCVRunnerImpl(
        dataset=cmu_database, estimator=one_vs_rest_mlp_gs, use_impostor_samples=True
    )
    results = two_class_mlp_with_hpo.exec()
    repo = results_repository_factory()
    repo.add_one_class_result(results, os.path.basename(__file__).replace(".py", ""))


if __name__ == "__main__":
    main()
