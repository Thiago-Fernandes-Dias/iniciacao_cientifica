import logging
import os

from sklearn.svm import OneClassSVM

from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import one_class_svm_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_global_hpo_runner import ExperimentWithGlobalHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithGlobalHPORunner(
            estimator_factory=lambda: OneClassSVM(),
            dataset=ds,
            params_grid=one_class_svm_params_grid,
            results_repo=results_repository_factory(),
            exp_name=str(os.path.basename(__file__).replace(".py", "")),
            use_impostor_samples=False
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
