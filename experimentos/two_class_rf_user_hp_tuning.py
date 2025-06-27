import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from lib.constants import RANDOM_STATE, N_JOBS
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import rf_params_grid
from lib.repositories.results_repository import ResultsRepository
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_with_search_cv_runner_impl import OneClassExperimentWithSearchCVRunnerImpl


def main() -> None:
    def est_factory(seed: int):
        two_class_rf_grid_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )
        two_class_rf_gs = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=rf_params_grid,
            cv=two_class_rf_grid_cv,
            n_jobs=N_JOBS,
            scoring="accuracy",
        )
        return two_class_rf_gs
    executor = ExperimentExecutor(
        results_repo=results_repository_factory(),
        name=str(os.path.basename(__file__).replace(".py", "")),
        runner_factory=lambda ds: OneClassExperimentWithSearchCVRunnerImpl(
            dataset=ds, 
            estimator_factory=est_factory,
            use_impostor_samples=True
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
