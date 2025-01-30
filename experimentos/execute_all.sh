#!/bin/bash

experiment_scripts=(
    multi_class_mlp_with_hpo.py
    multi_class_rf.py
    multi_class_rf_with_hpo.py
    multi_class_mlp.py
    one_class_mlp_with_hpo.py
    one_class_mlp.py
    one_class_svm_with_hpo.py
    one_class_svm.py
    one_vs_rest_rf_with_hpo.py
    one_vs_rest_rf.py
)

for script in "${experiment_scripts[@]}"
do
    echo "Executing $script"
    python $script
done