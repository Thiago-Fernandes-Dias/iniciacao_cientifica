import os

from sklearn.ensemble import RandomForestClassifier

from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl


def main() -> None:
    executor = ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""),
        runner_factory=lambda ds: OneClassExperimentRunnerImpl(
            dataset=ds,
            estimator=RandomForestClassifier(), 
            use_impostor_samples=True
        ),
        results_repo=results_repository_factory()
    )
    executor.execute()

if __name__ == "__main__":
    main()