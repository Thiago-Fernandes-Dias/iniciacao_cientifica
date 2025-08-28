import logging

from sklearn.ensemble import RandomForestClassifier

from lib.datasets.cmu_dataset import CMUDataset
from lib.datasets.keyrecs_dataset import KeyrecsDataset
from lib.hp_grids import rf_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.runners.experiment_with_global_hpo_runner import ExperimentWithGlobalHPORunner
from lib.utils import CMU_PATH, KEYRECS_PATH, cmu_default_split, keyrecs_default_split


def main() -> None:
    logging.basicConfig(level=logging.NOTSET)

    cmu = CMUDataset(CMU_PATH, cmu_default_split)
    cmu_runner = ExperimentWithGlobalHPORunner(
        estimator_factory=lambda s: RandomForestClassifier(random_state=s),
        dataset=cmu,
        params_grid=rf_params_grid,
        results_repo=results_repository_factory(),
        exp_name="Random Forest com HPO global (CMU)",
        use_impostor_samples=True,
        seeds_range=list(range(14, 30))
    )
    cmu_runner.add_name_suffix("CMU")
    cmu_runner.exec()

    keyrecs = KeyrecsDataset(KEYRECS_PATH, keyrecs_default_split)
    keyrecs_runner = ExperimentWithGlobalHPORunner(
        estimator_factory=lambda s: RandomForestClassifier(random_state=s),
        dataset=keyrecs,
        params_grid=rf_params_grid,
        results_repo=results_repository_factory(),
        exp_name="Random Forest com HPO global (Keyrecs)",
        use_impostor_samples=True,
        seeds_range=list(range(30))
    )
    keyrecs_runner.add_name_suffix("Keyrecs")
    keyrecs_runner.exec()


if __name__ == "__main__":
    main()
