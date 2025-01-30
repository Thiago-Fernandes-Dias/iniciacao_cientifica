from sklearn.ensemble import RandomForestClassifier

from cmu_dataset import CMUDataset
from runners.multi_class_experiment_runner import MultiClassExperimentRunner
from utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    multi_class_rf = MultiClassExperimentRunner(dataset=cmu_database, estimator=RandomForestClassifier())
    results = multi_class_rf.exec()
    save_results("multi_class_rf", results.to_dict())

if __name__ == "__main__":
    main()