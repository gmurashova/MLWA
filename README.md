# MLWA


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
