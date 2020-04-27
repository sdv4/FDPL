"""
    Creates HDF5 dataset file for super resolution task.
    note: only the luminance channel of the YCbCr image is saved.
"""
import os
import sys
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import h5py

if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "<path_to_images/>", "<train_or_test_or_val/file_name.hdf5>",
          "<super_resolution_factor>")
    """
    """
    sys.exit()

PATCH_SIZE = 32 #dimension of square patch size to be used for training
SR_FACTOR = int(sys.argv[3])
PATH = sys.argv[1]
img_names_list = [f for f in os.listdir(PATH) if f[-4:] == '.png' or f[-4:] == '.jpg']
hdf5_file = h5py.File('../../data/processed/' + sys.argv[2], "w")

training_patches = []
target_patches = []
print("Creating image/target pairs...")
for idx in tqdm(range(len(img_names_list))):
    img_name = img_names_list[idx]
    # load luminance channel
    ImageFile = Image.open(PATH + img_name)
    im = np.array(ImageFile.convert('YCbCr'), dtype=np.float)
    #crop image to be multiple of 8 in both dims
    max_x = int(im.shape[0]/8) * 8
    max_y = int(im.shape[1]/8) * 8
    square_dim = min(max_x, max_y)
    im_Y = im[0:square_dim, 0:square_dim, 0]
    im_Cb = im[0:square_dim, 0:square_dim, 1]
    im_Cr = im[0:square_dim, 0:square_dim, 2]

    # distort image with blur, downsize and upsize bicubic interpolation
    im_Y_blur = distort_image(path=PATH + img_name, 
                              factor=SR_FACTOR, 
                              sigma=SIGMA)[0:square_dim, 0:square_dim, 0].astype(np.float)

    for i in range(0, im_Y_blur.shape[1]-PATCH_SIZE, 13): #every patch vertically with stride 13
        for j in range(0, im_Y_blur.shape[0]-PATCH_SIZE, 13):
            sub_im_blur = im_Y_blur[j:j+PATCH_SIZE, i:i+PATCH_SIZE]
            sub_im = im_Y[j:j+PATCH_SIZE, i:i+PATCH_SIZE]
            training_patches.append(sub_im_blur)
            target_patches.append(sub_im)

data_shape = (len(training_patches), PATCH_SIZE, PATCH_SIZE)
hdf5_file.create_dataset("blurred_img", data_shape, np.float)
hdf5_file.create_dataset("target_img", data_shape, np.float)

print("Building HDF5 dataset...")
for i in tqdm(range(len(training_patches))):
    hdf5_file["blurred_img"][i, ...] = training_patches[i]
    hdf5_file["target_img"][i, ...] = target_patches[i]

hdf5_file.close()

print("Done creating %d training pairs from %d original images" %
      (len(target_patches), len(img_names_list)))
