import os

from lib.datasets.dataset import Dataset
from lib.estimators.improved_statistical_alg import ImprovedStatisticalAlg
from lib.experiment_executor import ExperimentExecutor
from lib.global_hp_search import runner_with_global_hpo_factory
from lib.hp_grids import st_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_runner import ExperimentRunner


def runner_fac(ds: Dataset) -> ExperimentRunner:
    exp_name = str(os.path.basename(__file__).replace(".py", ""))
    return runner_with_global_hpo_factory(
        ds=ds,
        estimator_factory=lambda: ImprovedStatisticalAlg(),
        params_grid=st_params_grid,
        repo=results_repository_factory(),
        exp_name=exp_name,
        use_impostor_samples=False
    )


def main() -> None:
    executor = ExperimentExecutor(runner_factory=runner_fac)
    executor.execute()


if __name__ == "__main__":
    main()
