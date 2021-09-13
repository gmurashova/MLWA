#!/bin/bash

for sz in 64 85 128 256
do
	mkdir ./TILES_${sz}/
	python ./Data_Generator/ML_Image_Tiler.py ${sz} ./Single_PMT_Images_For_Image_Classifier/  
	python ./Data_Generator/Sort_MMI_Files_by_Category.py ./TILES_${sz}/
	python ./Data_Generator/create_label_vector.py ./TILES_${sz}/ 
done

for sz in 64 85 128 256
do
	mkdir ./TILES_${sz}_AR_Corrected/
        python ./Data_Generator/ML_Image_Tiler.py ${sz} ./Single_PMT_Images_For_Image_Classifier_AR_Corrected/ AR_Corrected 
        python ./Data_Generator/Sort_MMI_Files_by_Category.py ./TILES_${sz}_AR_Corrected/
        python ./Data_Generator/create_label_vector.py ./TILES_${sz}_AR_Corrected/
done




