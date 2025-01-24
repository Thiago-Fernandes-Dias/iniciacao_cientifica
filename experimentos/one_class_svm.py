from sklearn.svm import OneClassSVM

from cmu_dataset import CMUDataset
from runners.single_class_experiment_runner import SingleClassExperimentRunner
from utils import first_session_split, save_results

def main() -> None:
    cmu_database = CMUDataset('datasets/cmu/DSL-StrongPasswordData.csv', first_session_split)
    one_class_svm_experiment = SingleClassExperimentRunner(dataset=cmu_database, estimator=OneClassSVM())
    results = one_class_svm_experiment.exec()
    save_results("one_class_svm", results.to_dict())

if __name__ == "__main__":
    main()