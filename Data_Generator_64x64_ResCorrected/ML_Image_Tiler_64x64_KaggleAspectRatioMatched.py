#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
# Helper libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('agg')
import numpy as np 
import pandas as pd 
import glob, os
from glob import iglob
import image_slicer
from pathlib import Path
import fnmatch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image

#####################################################################################################
############################################## STEP 1 ############################################### 
##################################################################################################### 
### Going through all of the subdirectories in the Single_PMT_Images_For_Image_Classifier folder and searching for the 512x512 HOSCC (human) JPG files

image_folders = '/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/Single_PMT_Images_For_Image_Classifier/'
pattern = '*HOSCC*'
for root, dirs, files in os.walk(image_folders):
    for name in files:
        if fnmatch.filter(files,pattern) and name.endswith((".jpg", ".png")):
            print(name)
            ### Giving each of the found image files their direct path name so that they can be referenced 
jpgfiles = [os.path.join(root, name)
                         for root, dirs, files in os.walk(image_folders)
                         for name in files
                         if fnmatch.filter(files,pattern) and name.endswith((".jpg", ".png"))]
                         
#####Correcting the Aspect Ratio to match the Kaggle Resolution and saving them to a different directory
for name in jpgfiles:
    name2 = Path(name).stem
    image_file = Image.open(name)
    width_org, height_org = image_file.size
    factor = 0.75 ## ideally would be 0.79
    print("Original Width and Height (pixel x pixel):", width_org, height_org)
    width = int(width_org * factor)
    height = int(height_org * factor)
    print("New Width and Height (pixel x pixel):", width, height)
    new_im = image_file.resize((width, height), Image.ANTIALIAS)
    new_im.save('/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/Single_PMT_Images_For_Image_Classifier_AR_Corrected/'+name2+'.png')

#####################################################################################################
############################################## STEP 2 ###############################################  
#####################################################################################################   
    
### Going through all of the subdirectories in the Single_PMT_Images_For_Image_Classifier folder and searching for the 512x512 HOSCC (human) JPG files
AR_image_folders = '/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/Single_PMT_Images_For_Image_Classifier_AR_Corrected/'
pattern = '*HOSCC*'
for root, dirs, files in os.walk(AR_image_folders):
    for name in files:
        if fnmatch.filter(files,pattern) and name.endswith((".png")):
            print(name)
            ### Giving each of the found image files their direct path name so that they can be referenced 
            
# print(jpgfiles)
          
ARC_files = [os.path.join(root, name)
                         for root, dirs, files in os.walk(AR_image_folders)
                         for name in files
                         if fnmatch.filter(files,pattern) and name.endswith((".jpg", ".png"))]
                         
## slicing each of those  512x512pixel images into 8 pieces so they will be 64 pixels                           
i = 0
for name in ARC_files:
    name2 = Path(name).stem
    print(name2)
    i += 1
    name3 = str(i)
    tiles = image_slicer.slice(name, 36, save=False) ### dimmensions/image size = # of images, square # of images to put into parameters
    image_slicer.save_tiles(tiles, directory='/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_64_AR_Corrected',prefix=name2+'_'+name3+'_slice', format='png')

    
