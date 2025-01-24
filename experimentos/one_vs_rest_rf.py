from sklearn.ensemble import RandomForestClassifier

from cmu_dataset import CMUDataset
from runners.one_vs_rest_experiment_runner import OneVsRestExperimentRunner
from utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    exp = OneVsRestExperimentRunner(cmu_database=cmu_database, estimator=RandomForestClassifier())
    results = exp.exec()
    save_results("two_class_rf", results.to_dict())

if __name__ == "__main__":
    main()