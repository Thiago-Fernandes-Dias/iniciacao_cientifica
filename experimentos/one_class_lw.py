import os

from lib.cmu_dataset import CMUDataset
from lib.lightweight_alg import LightWeightAlg
from lib.runners.single_class_experiment_runner import SingleClassExperimentRunner
from lib.utils import lw_split, first_session_split, save_results


def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv',
                              first_session_split, "(DD|UD)(\\.[A-Za-z]+)+")
    one_class_lw_experiment = SingleClassExperimentRunner(dataset=cmu_database, estimator=LightWeightAlg())
    results = one_class_lw_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())


if __name__ == "__main__":
    main()
