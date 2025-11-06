import logging
import os

from lib.estimators.improved_statistical_alg import ImprovedStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import st_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_user_hpo_runner import ExperimentWithUserHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    name = str(os.path.basename(__file__).replace(".py", ""))
    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithUserHPORunner(
            dataset=ds, estimator_factory=lambda s: ImprovedStatisticalAlg(),
            params_grid=st_params_grid,
            results_repo=results_repository_factory(),
            exp_name="Magalhães com HPO por usuário",
            use_impostor_samples=False,
            seeds_range=range(0, 5),
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
