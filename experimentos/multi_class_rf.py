import os
from sklearn.ensemble import RandomForestClassifier

from lib.datasets.dataset import Dataset
from lib.runners.multi_class_experiment_runner import MultiClassExperimentRunner
from lib.utils import cmu_first_session_split, save_results

def main() -> None:
    cmu_database = Dataset('datasets/cmu/DSL-StrongPasswordData.csv', cmu_first_session_split)
    multi_class_rf = MultiClassExperimentRunner(dataset=cmu_database, estimator=RandomForestClassifier())
    results = multi_class_rf.exec()
    save_results(os.path.basename(__file__).replace(".py", ""), results.to_dict())

if __name__ == "__main__":
    main()