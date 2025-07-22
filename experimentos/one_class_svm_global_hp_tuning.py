import os

from sklearn.svm import OneClassSVM

from lib.datasets.dataset import Dataset
from lib.experiment_executor import ExperimentExecutor
from lib.global_hp_search import runner_with_global_hpo_factory
from lib.hp_grids import one_class_svm_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_runner import ExperimentRunner


def runner_fac(ds: Dataset) -> ExperimentRunner:
    name = str(os.path.basename(__file__).replace(".py", ""))
    return runner_with_global_hpo_factory(
        ds=ds, estimator_factory=lambda: OneClassSVM(),
        params_grid=one_class_svm_params_grid,
        repo=results_repository_factory(), exp_name=name,
        use_impostor_samples=False
    )


def main() -> None:
    executor = ExperimentExecutor(runner_factory=runner_fac)
    executor.execute()


if __name__ == "__main__":
    main()
