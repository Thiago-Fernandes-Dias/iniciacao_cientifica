import os
from sklearn.ensemble import RandomForestClassifier

from lib.cmu_dataset import CMUDataset
from lib.runners.multi_class_experiment_runner import MultiClassExperimentRunner
from lib.utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    multi_class_rf = MultiClassExperimentRunner(dataset=cmu_database, estimator=RandomForestClassifier())
    results = multi_class_rf.exec()
    save_results(os.path.basename(__file__), results.to_dict())

if __name__ == "__main__":
    main()