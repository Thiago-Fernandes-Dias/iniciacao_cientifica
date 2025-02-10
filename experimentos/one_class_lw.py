import os

from lib.cmu_dataset import CMUDataset
from lib.lightweight_alg import LightWeightAlg
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl
from lib.utils import lw_split, save_results


def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv',
                              lw_split, "(DD|UD)(\\.[A-Za-z]+)+")
    one_class_lw_experiment = OneClassExperimentRunnerImpl(dataset=cmu_database, estimator=LightWeightAlg())
    results = one_class_lw_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())


if __name__ == "__main__":
    main()
