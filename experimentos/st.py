import logging
import os

from lib.estimators.improved_statistical_alg import ImprovedStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithoutHPORunner(
            results_repo=results_repository_factory(),
            dataset=ds, estimator_factory=lambda s: ImprovedStatisticalAlg(),
            exp_name="Magalhães",
            use_impostor_samples=False
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
