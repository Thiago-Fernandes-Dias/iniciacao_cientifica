import logging
import os

from sklearn.ensemble import RandomForestClassifier

from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import rf_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_global_hpo_runner import ExperimentWithGlobalHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithGlobalHPORunner(
            estimator_factory=lambda s: RandomForestClassifier(random_state=s),
            dataset=ds,
            params_grid=rf_params_grid,
            results_repo=results_repository_factory(),
            exp_name="Random Forest com HPO global",
            use_impostor_samples=True
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
