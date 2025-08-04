import logging
import os

from sklearn.ensemble import RandomForestClassifier

from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import rf_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_user_hpo_runner import ExperimentWithUserHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    name = str(os.path.basename(__file__).replace(".py", ""))
    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithUserHPORunner(
            params_grid=rf_params_grid,
            dataset=ds,
            estimator_factory=lambda: RandomForestClassifier(),
            exp_name=name,
            results_repo=results_repository_factory(),
            use_impostor_samples=True
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
