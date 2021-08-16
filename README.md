# MLWA

## Create Conda Environment

Create the conda environment to run the experiments.  

```conda env create --prefix ./envs --file environment.yml```

or

```makefile```


## Downloading Kaggle Dataset

1. Create a directory called colorectal-histology-mnist using the following command: ```mkdir colorectal-histology-mnist```
2. Generate a kaggle login and then go to the [colorectal histology mnist](https://www.kaggle.com/kmader/colorectal-histology-mnist) to download the required datase. Save the ```archive.zip``` file in the above directory.
3. Extract the file using the ofllowing command: ```unzip archive.zip```


## Generating NMMMI Image Data 

There are three scripts in each "Data_Generator_*" folder. The scripts must be run in the following order:

1. ML_Image_Tiler_Dim1xDim2.py <-- This file goes through all of the original NMMMI images in the designated folder (either original resolution or the artificially corrected resolution (AR)) and tiles them based on the output tile dimensions specified. Make sure the path is going to the appropriate image folder. If you are tiling the original resolution NMMMIs, use this folder: "Single_PMT_Images_For_Image_Classifier", if you are tiling the artificially corrected resolution NMMMIs, use this folder: "Single_PMT_Images_For_Image_Classifier_AR_Corrected"

2. Sorting_MMI_tiles.py <- this file organizes the newly-tiled NMMMIs into the appropriate class folder that it creates, e.g. CANCER, INFLAMMATORY, or HEALTHY.

3. create_label_vector.py <-- this file creates an array of labels for the tiles that are one-hot encoded so the ML program can import, use, and interpret them.

## Review trials

1. ML Hyperparam Trials <-- this is the folder with the scripts to run the hyperparameter trials, where I test the various learning rates and optimizer functions on the Kaggle dataset.
2. ML_Kaggle_TL <-- this contains the transfer learning "Kaggle" model scripts and data (colorectal-histology-mnist is the folder with the MNIST data from Kaggle)
3. ML_MMI_TL <-- this folder contains all of the python scripts for the transfer learning portion of the work. each script adjusts the layer in the previously-trained model where we insert new data and begin retraining.
4. ML_Research_64_NoARC <-- this is the folder containing the script where we are using 64x64 MMI tiles WITHOUT adjusting the resolution to match the Kaggle images (hence, "NoARC", No artificial resolution correction)
5. ML_Research_Kaggle <-- this contains the initial "Kaggle" model scripts and data (colorectal-histology-mnist is the folder with the MNIST data from Kaggle)
6. ML_Research_MMI <-- this contains the scripts for the model training on the 64x64 NMMMI images with the artificial resolution correction (ARC) to "match" the Kaggle image resolution
7. ML_Research_MMI_128_ARC <-- this contains the scripts to train the model on 128x128 NMMMI tiles with ARC
8. ML_Research_MMI_128_NoAug <-- this contains the scripts to train the model on 128x128 NMMMI tiles without ARC
9. ML_Research_MMI_256 <-- this contains the scripts to train the model on 256x256 NMMMI tiles without ARC
10. ML_Research_MMI_256_ARC <-- this contains the scripts to train the model on 256x256 NMMMI tiles with ARC
11. ML_Research_MMI_85 <-- this contains the scripts to train the model on 85x85 NMMMI tiles without ARC
