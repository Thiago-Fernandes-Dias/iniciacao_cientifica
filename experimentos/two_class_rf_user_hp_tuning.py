import os

from sklearn.ensemble import RandomForestClassifier

from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import rf_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_two_classes_hpo_runner import ExperimentWithTwoClassesRunner


def main() -> None:
    name = str(os.path.basename(__file__).replace(".py", ""))
    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithTwoClassesRunner(
            params_grid=rf_params_grid,
            dataset=ds,
            estimator_factory=lambda: RandomForestClassifier(),
            exp_name=name,
            results_repo=results_repository_factory()
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
