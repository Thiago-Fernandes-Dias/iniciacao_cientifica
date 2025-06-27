import os

from sklearn.model_selection import KFold, ParameterGrid

from lib.constants import RANDOM_STATE
from lib.datasets.dataset import Dataset
from lib.estimators.improved_statistical_alg import ImprovedStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.global_hp_search import GlobalHPTuning
from lib.hp_grids import st_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl


def runner_with_optimized_threshold_factory(dataset: Dataset) -> OneClassExperimentRunnerImpl:
    cv = KFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
    threshold_tuning = GlobalHPTuning(
        dataset=dataset,
        params_grid=ParameterGrid(st_params_grid),
        estimator_factory=lambda: ImprovedStatisticalAlg(),
        cv=cv
    )
    best_params_config = threshold_tuning.search()
    return OneClassExperimentRunnerImpl(
        dataset=dataset,
        estimator=ImprovedStatisticalAlg(threshold=best_params_config['threshold']),
        use_impostor_samples=False
    )


def main() -> None:
    executor = ExperimentExecutor(
        name=str(os.path.basename(__file__).replace(".py", "")),
        results_repo=results_repository_factory(),
        runner_factory=runner_with_optimized_threshold_factory
    )
    executor.execute()


if __name__ == "__main__":
    main()
