import os

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import OneClassSVM

from lib.constants import N_JOBS
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import one_class_svm_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_with_search_cv_runner_impl import OneClassExperimentWithSearchCVRunnerImpl


def main() -> None:
    one_class_svm_grid_cv = KFold(n_splits=5)
    one_class_svm_gs = GridSearchCV(
        OneClassSVM(),
        one_class_svm_params_grid,
        scoring="accuracy",
        cv=one_class_svm_grid_cv,
        n_jobs=N_JOBS,
    )
    executor = ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""),
        results_repo=results_repository_factory(),
        runner_factory=lambda ds: OneClassExperimentWithSearchCVRunnerImpl(
            dataset=ds, estimator=one_class_svm_gs
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
