import os

from sklearn.neural_network import MLPClassifier

from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository import results_repository_factory
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl


def main() -> None:
    executor = ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""),
        runner_factory=lambda ds: OneClassExperimentRunnerImpl(
            dataset=ds, 
            estimator=MLPClassifier(hidden_layer_sizes=(100, 400, 100)), 
            use_impostor_samples=True
        ),
        results_repo=results_repository_factory()
    )
    executor.execute()


if __name__ == "__main__":
    main()
