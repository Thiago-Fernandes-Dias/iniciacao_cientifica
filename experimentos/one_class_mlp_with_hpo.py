import os

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier

from lib.constants import N_JOBS
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import mlp_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_with_search_cv_runner_impl import OneClassExperimentWithSearchCVRunnerImpl


def main() -> None:
    one_class_mlp_grid_cv = KFold(n_splits=5)
    one_class_mlp_gs = GridSearchCV(
        MLPClassifier(),
        mlp_params_grid,
        scoring="accuracy",
        cv=one_class_mlp_grid_cv,
        n_jobs=N_JOBS,
    )
    executor = ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""),
        results_repo=results_repository_factory(),
        runner_factory=lambda ds: OneClassExperimentWithSearchCVRunnerImpl(
            dataset=ds, estimator_factory=one_class_mlp_gs
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
