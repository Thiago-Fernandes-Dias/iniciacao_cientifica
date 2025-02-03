#!/bin/bash

for script in `ls *.py`
do
    echo "Executing $script"
    python $script
done