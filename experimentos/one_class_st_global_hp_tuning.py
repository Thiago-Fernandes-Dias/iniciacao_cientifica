import logging
import os

from lib.estimators.improved_statistical_alg import ImprovedStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import st_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_global_hpo_runner import ExperimentWithGlobalHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithGlobalHPORunner(
            estimator_factory=lambda: ImprovedStatisticalAlg(),
            dataset=ds,
            params_grid=st_params_grid,
            results_repo=results_repository_factory(),
            exp_name=str(os.path.basename(__file__).replace(".py", "")),
            use_impostor_samples=False
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
