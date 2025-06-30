import os

from lib.estimators.improved_statistical_alg import ImprovedStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import st_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_one_class_hpo_runner import ExperimentWithOneClassHPORunner


def main() -> None:
    executor = ExperimentExecutor(
        name=str(os.path.basename(__file__).replace(".py", "")),
        results_repo=results_repository_factory(),
        runner_factory=lambda ds: ExperimentWithOneClassHPORunner(
            dataset=ds, estimator_factory=lambda: ImprovedStatisticalAlg(), params_grid=st_params_grid,
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
