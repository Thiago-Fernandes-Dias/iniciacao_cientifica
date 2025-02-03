import os
from sklearn.neural_network import MLPClassifier

from lib.cmu_dataset import CMUDataset
from lib.runners.one_vs_rest_experiment_runner import OneVsRestExperimentRunner
from lib.utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    exp = OneVsRestExperimentRunner(cmu_database=cmu_database, estimator=MLPClassifier())
    results = exp.exec()
    save_results(os.path.basename(__file__), results.to_dict())

if __name__ == "__main__":
    main()