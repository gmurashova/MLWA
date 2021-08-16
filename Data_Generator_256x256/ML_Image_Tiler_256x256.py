#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
#
# # TensorFlow and tf.keras
# import tensorflow as tf
# from tensorflow import keras
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


### Going through all of the subdirectories in the Single_PMT_Images_For_Image_Classifier folder and searching for the 512x512 HOSCC (human) JPG files

image_folders = '/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/Single_PMT_Images_For_Image_Classifier/'
pattern = '*HOSCC*'
for root, dirs, files in os.walk(image_folders):
    for name in files:
        if fnmatch.filter(files,pattern) and name.endswith((".jpg",".png")):
            print(name)
            ### Giving each of the found image files their direct path name so that they can be referenced 
jpgfiles = [os.path.join(root, name)
                         for root, dirs, files in os.walk(image_folders)
                         for name in files
                         if fnmatch.filter(files,pattern) and name.endswith((".jpg",".png"))]
# print(jpgfiles)
## slicing each of those  512x512pixel images into 8 pieces so they will be 64 pixels
i = 0
for name in jpgfiles:
    name2 = Path(name).stem
    print(name2)
    i += 1
    name3 = str(i)
    tiles = image_slicer.slice(name, 4, save=False)### dimmensions/image size = # of images, square # of images to put into
    image_slicer.save_tiles(tiles, directory='/Users/gabrielleosborn-lipsitz/MOVE_BACK_TO_DESKTOP/ML_Research/TILES_256',prefix=name2+'_'+name3+'_slice', format='png')

