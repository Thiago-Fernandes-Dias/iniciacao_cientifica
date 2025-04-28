import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from lib.constants import RANDOM_STATE, N_JOBS
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import rf_params_grid
from lib.runners.one_class_experiment_with_search_cv_runner_impl import OneClassExperimentWithSearchCVRunnerImpl


def main() -> None:
    two_class_rf_grid_cv = StratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=RANDOM_STATE
    )
    two_class_rf_gs = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=rf_params_grid,
        cv=two_class_rf_grid_cv,
        n_jobs=N_JOBS,
        scoring="accuracy",
    )
    executor = ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""),
        runner_factory=lambda ds: OneClassExperimentWithSearchCVRunnerImpl(
            dataset=ds, 
            estimator=two_class_rf_gs, 
            use_impostor_samples=True
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
