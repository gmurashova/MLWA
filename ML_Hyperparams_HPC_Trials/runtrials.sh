#!/bin/bash

mkdir Figures
mkdir Models

./GM_ML_Histo_new_from_tutorial_op.py adadelta 0.0001
./GM_ML_Histo_new_from_tutorial_op.py adadelta 0.001
./GM_ML_Histo_new_from_tutorial_op.py adadelta 0.002
./GM_ML_Histo_new_from_tutorial_op.py adadelta 0.01
./GM_ML_Histo_new_from_tutorial_op.py adagradi 0.0001
./GM_ML_Histo_new_from_tutorial_op.py adagrad 0.001
./GM_ML_Histo_new_from_tutorial_op.py adagrad 0.002
./GM_ML_Histo_new_from_tutorial_op.py adagrad 0.01
./GM_ML_Histo_new_from_tutorial_op.py adamax 0.0001
./GM_ML_Histo_new_from_tutorial_op.py adamax 0.001
./GM_ML_Histo_new_from_tutorial_op.py adamax 0.002
./GM_ML_Histo_new_from_tutorial_op.py adamax 0.01
./GM_ML_Histo_new_from_tutorial_op.py adam 0.0001
./GM_ML_Histo_new_from_tutorial_op.py adam 0.001
./GM_ML_Histo_new_from_tutorial_op.py adam 0.002
./GM_ML_Histo_new_from_tutorial_op.py adam 0.01
./GM_ML_Histo_new_from_tutorial_op.py nadam 0.0001
./GM_ML_Histo_new_from_tutorial_op.py nadam 0.001
./GM_ML_Histo_new_from_tutorial_op.py nadam 0.002
./GM_ML_Histo_new_from_tutorial_op.py nadam 0.01


