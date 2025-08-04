import logging
import os

from sklearn.svm import OneClassSVM

from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    results_repo = results_repository_factory()
    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithoutHPORunner(
            dataset=ds,
            estimator=OneClassSVM(),
            exp_name="SVM",
            results_repo=results_repo,
            use_impostor_samples=False
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
