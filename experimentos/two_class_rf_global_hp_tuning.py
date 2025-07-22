import os

from sklearn.ensemble import RandomForestClassifier

from lib.experiment_executor import ExperimentExecutor
from lib.hp_grids import rf_params_grid
from lib.repositories.results_repository_factory import results_repository_factory
from lib.global_hp_search import runner_with_global_hpo_factory


def main() -> None:
    name = str(os.path.basename(__file__).replace(".py", ""))
    executor = ExperimentExecutor(
        runner_factory=lambda ds: runner_with_global_hpo_factory(params_grid=rf_params_grid,
                                                                 estimator_factory=lambda: RandomForestClassifier(),
                                                                 ds=ds, exp_name=name, 
                                                                 repo=results_repository_factory(),
                                                                 use_impostor_samples=True),
    )
    executor.execute()


if __name__ == "__main__":
    main()
