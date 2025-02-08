import os

from sklearn.neural_network import MLPClassifier

from lib.cmu_dataset import CMUDataset
from lib.runners.one_class_experiment_runner_impl import OneClassExperimentRunnerImpl
from lib.utils import first_session_split, save_results


def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    exp = OneClassExperimentRunnerImpl(dataset=cmu_database, estimator=MLPClassifier(), use_impostor_samples=True)
    results = exp.exec()
    save_results(os.path.basename(__file__), results.to_dict())


if __name__ == "__main__":
    main()
