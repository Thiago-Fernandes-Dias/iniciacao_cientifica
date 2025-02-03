import os
from sklearn.neural_network import MLPClassifier

from lib.cmu_dataset import CMUDataset
from lib.runners.multi_class_experiment_runner import MultiClassExperimentRunner
from lib.utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    multi_class_mlp = MultiClassExperimentRunner(dataset=cmu_database, estimator=MLPClassifier())
    results = multi_class_mlp.exec()
    save_results(os.path.basename(__file__), results.to_dict())

if __name__ == "__main__":
    main()