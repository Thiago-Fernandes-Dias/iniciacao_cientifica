import logging

from sklearn.ensemble import RandomForestClassifier

from lib.experiment_executor import ExperimentExecutor
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_without_hpo_runner import ExperimentWithoutHPORunner
from lib.utils import N_JOBS


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithoutHPORunner(
            exp_name="Random Forest",
            dataset=ds,
            estimator_factory=lambda s: RandomForestClassifier(n_jobs=N_JOBS, random_state=s),
            use_impostor_samples=True,
            results_repo=results_repository_factory()
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
