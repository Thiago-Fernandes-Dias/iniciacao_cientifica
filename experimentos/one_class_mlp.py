import os
from sklearn.neural_network import MLPClassifier

from lib.cmu_dataset import CMUDataset
from lib.runners.single_class_experiment_runner import SingleClassExperimentRunner
from lib.utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    one_class_svm_experiment = SingleClassExperimentRunner(dataset=cmu_database, estimator=MLPClassifier())
    results = one_class_svm_experiment.exec()
    save_results(os.path.basename(__file__), results.to_dict())

if __name__ == "__main__":
    main()