import logging

from sklearn.ensemble import RandomForestClassifier

from lib.datasets.keyrecs_dataset import KeyrecsDataset
from lib.hp_grids import rf_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_user_hpo_runner import ExperimentWithUserHPORunner
from lib.utils import KEYRECS_PATH, keyrecs_default_split


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    keyrecs = KeyrecsDataset(KEYRECS_PATH, keyrecs_default_split)
    keyrecs_runner = ExperimentWithUserHPORunner(
        estimator_factory=lambda s: RandomForestClassifier(random_state=s),
        dataset=keyrecs,
        params_grid=rf_params_grid,
        results_repo=results_repository_factory(),
        exp_name="Random Forest com HPO global (Keyrecs)",
        use_impostor_samples=True,
        seeds_range=list(range(30))
    )
    keyrecs_runner.exec()


if __name__ == "__main__":
    main()
