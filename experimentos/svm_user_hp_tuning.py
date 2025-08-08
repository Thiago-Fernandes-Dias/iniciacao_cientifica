import logging
import os

from sklearn.svm import OneClassSVM

from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import one_class_svm_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_user_hpo_runner import ExperimentWithUserHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)
    
    results_repo = results_repository_factory()
    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithUserHPORunner(
            dataset=ds, estimator_factory=lambda: OneClassSVM(),
            params_grid=one_class_svm_params_grid,
            results_repo=results_repo,
            exp_name="SVM com HPO por usuário",
            use_impostor_samples=False
        )
    )
    executor.execute()


if __name__ == "__main__":
    main()
