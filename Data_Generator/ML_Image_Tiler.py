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

### Command Line Arguments
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

image_folders = '../Single_PMT_Images_For_Image_Classifier/'
tile_sz=128

if len(sys.argv) > 1:
    tile_sz=int(sys.argv[1]) 

if len(sys.argv) > 2:
    image_folders = sys.argv[2]

out_folder = f'./TILES_{tile_sz}'
if len(sys.argv) > 3:
    out_folder = f'./TILES_{tile_sz}_{sys.argv[3]}'

print(f"{tile_sz=}")
print(f"{image_folders=}")
print(f"{out_folder=}")

### End input argument code ###

im_sz=512
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
print(jpgfiles)
## slicing each of those  512x512pixel images into 8 pieces so they will be 64 pixels
i = 0
for name in jpgfiles:
    name2 = Path(name).stem
    print(name2)
    i += 1
    name3 = str(i)
    num_subimgs = int(im_sz/tile_sz*im_sz/tile_sz)
    print(f"{num_subimgs=}")
    tiles = image_slicer.slice(name, num_subimgs, save=False)
    image_slicer.save_tiles(tiles, directory=out_folder,prefix=f"{name2}_{name3}_slice", format='png')


