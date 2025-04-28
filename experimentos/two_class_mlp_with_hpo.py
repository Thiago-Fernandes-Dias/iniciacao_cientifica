import os

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier

from lib.constants import RANDOM_STATE, N_JOBS
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import mlp_params_grid
from lib.repositories.results_repository import results_repository_factory
from lib.runners.one_class_experiment_with_search_cv_runner_impl import OneClassExperimentWithSearchCVRunnerImpl


def main() -> None:
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
    executor=ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""),
        runner_factory=lambda ds: OneClassExperimentWithSearchCVRunnerImpl(
            dataset=ds, estimator=one_vs_rest_mlp_gs, use_impostor_samples=True
        ),
        results_repo=results_repository_factory()
    )
    executor.execute()


if __name__ == "__main__":
    main()
