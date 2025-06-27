import os

from sklearn.model_selection import KFold

from lib.constants import RANDOM_STATE
from lib.datasets.dataset import Dataset
from lib.experiment_executor import ExperimentExecutor
from lib.global_threshold_search import GlobalThresholdTuning
from lib.estimators.improved_statistical_alg import ImprovedStatisticalAlg
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl
from lib.utils import float_range


def runner_with_optimized_threshold_factory(dataset: Dataset) -> OneClassExperimentRunnerImpl:
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    threshold_tuning = GlobalThresholdTuning(
        dataset=dataset,
        thresholds=float_range(0.2, 0.9, 0.05),
        estimator_factory=lambda: ImprovedStatisticalAlg(),
        cv=cv
    )
    best_threshold = threshold_tuning.search()
    return OneClassExperimentRunnerImpl(
        dataset=dataset, 
        estimator=ImprovedStatisticalAlg(threshold=best_threshold),
        use_impostor_samples=False
    )


def main() -> None:
    executor = ExperimentExecutor(
        name=os.path.basename(__file__).replace(".py", ""), 
        results_repo=results_repository_factory(),
        runner_factory=runner_with_optimized_threshold_factory
    )
    executor.execute()


if __name__ == "__main__":
    main()
