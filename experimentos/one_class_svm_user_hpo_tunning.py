import os

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.svm import OneClassSVM

from lib.constants import N_JOBS
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import one_class_svm_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_with_search_cv_runner_impl import OneClassExperimentWithSearchCVRunnerImpl


def main() -> None:
    def est_fac(seed: int) -> BaseSearchCV:
        one_class_svm_grid_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        one_class_svm_gs = GridSearchCV(
            OneClassSVM(),
            one_class_svm_params_grid,
            scoring="accuracy",
            cv=one_class_svm_grid_cv,
            n_jobs=N_JOBS,
        )
        return one_class_svm_gs
    executor = ExperimentExecutor(
        name=str(os.path.basename(__file__).replace(".py", "")),
        results_repo=results_repository_factory(),
        runner_factory=lambda ds: OneClassExperimentWithSearchCVRunnerImpl(
            dataset=ds, estimator_factory=est_fac,
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
