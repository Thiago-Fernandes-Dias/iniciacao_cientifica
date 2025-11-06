import logging

from sklearn.ensemble import RandomForestClassifier

from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import rf_params_grid
from lib.utils import default_seeds_range
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_user_hpo_runner import ExperimentWithUserHPORunner


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    executor = ExperimentExecutor(
        runner_factory=lambda ds: ExperimentWithUserHPORunner(
            params_grid=rf_params_grid,
            dataset=ds,
            estimator_factory=lambda s: RandomForestClassifier(random_state=s),
            exp_name="Random Forest com HPO por usuário",
            results_repo=results_repository_factory(),
            use_impostor_samples=True,
            seeds_range=[4],
        ),
    )
    executor.execute()


if __name__ == "__main__":
    main()
