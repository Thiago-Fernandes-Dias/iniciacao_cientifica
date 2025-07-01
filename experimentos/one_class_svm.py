import os

from sklearn.svm import OneClassSVM

from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner


def main() -> None:
    executor = ExperimentExecutor(
        name=str(os.path.basename(__file__).replace(".py", "")),
        results_repo=results_repository_factory(),
        runner_factory=lambda ds: ExperimentWithoutHPORunner(
            dataset=ds,
            estimator=OneClassSVM(),
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
