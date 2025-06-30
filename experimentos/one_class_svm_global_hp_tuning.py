import os

from sklearn.svm import OneClassSVM

from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import one_class_svm_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.global_hp_search import runner_with_global_hpo_factory


def main() -> None:
    executor = ExperimentExecutor(
        name=str(os.path.basename(__file__).replace(".py", "")),
        results_repo=results_repository_factory(),
        runner_factory=lambda ds: runner_with_global_hpo_factory(ds=ds, estimator_factory=lambda: OneClassSVM(),
                                                                 params_grid=one_class_svm_params_grid),
    )
    executor.execute()


if __name__ == "__main__":
    main()
