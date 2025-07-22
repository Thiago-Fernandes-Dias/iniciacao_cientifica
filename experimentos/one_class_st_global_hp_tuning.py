import os

from lib.estimators.improved_statistical_alg import ImprovedStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import st_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.global_hp_search import runner_with_global_hpo_factory


def main() -> None:
    exp_name = str(os.path.basename(__file__).replace(".py", ""))
    executor = ExperimentExecutor(
        runner_factory=lambda ds: runner_with_global_hpo_factory(ds=ds,
                                                                 estimator_factory=lambda: ImprovedStatisticalAlg(),
                                                                 params_grid=st_params_grid,
                                                                 repo=results_repository_factory(),
                                                                 exp_name=exp_name),
    )
    executor.execute()


if __name__ == "__main__":
    main()
