"""
    Functions used for distorting images for training
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import torch

from src.utils import dct_2d
from src.utils import idct_2d
from src.utils import ycbcr2rgb

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def distort_image(path, factor, sigma=1, blur=True):
    """ Distorts image by bluring it, decreasing its resolution
        by some factor, then increasing resolution - by bicubic
        interpolation.

        Args:
            path (string): absolute path to an image file
            factor (int): the resolution factor for interpolation
            sigma (float): the std. dev. to use for the gaussian blur
            blur (boolean): if True, gaussian blur is performed on im
        Returns:
            blurred_img (numpy.ndarray): distorted image in YCbCr with
                type uint8

    """
    image_file = Image.open(path)
    im = np.array(image_file.convert('YCbCr'))
    im_Y, im_Cb, im_Cr = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    im_Y = (im_Y.astype(np.int16)).astype(np.int64)
    im_Cb = (im_Cb.astype(np.int16)).astype(np.int64)
    im_Cr = (im_Cr.astype(np.int16)).astype(np.int64)
    if blur:
        im_Y_blurred = gaussian_filter(im_Y, sigma=sigma)
    else:
        im_Y_blurred = im_Y
    im_blurred = np.copy(im)
    im_blurred[:, :, 0] = im_Y_blurred
    im_blurred[:, :, 1] = im_Cb
    im_blurred[:, :, 2] = im_Cr
    width, length = im_Y.shape
    im_blurred = Image.fromarray(im_blurred, mode='YCbCr')
    im_blurred = im_blurred.resize(size=(int(length/factor),
                                         int(width/factor)),
                                   resample=Image.BICUBIC)

    im_blurred = im_blurred.resize(size=(length, width),
                                   resample=Image.BICUBIC)
    im_blurred = np.array(im_blurred.convert('YCbCr'))
    return im_blurred

def jpeg_distort(img):
    """
        Takes an image and applies quantization steps of JPEG
        encoding and returns the image with JPEG artifacts.
        NOTE: compression done only on Y/luminance channel
        Args:
            img (numpy array): YCbCr image in the form of a 3D array
        Returns:
            compressed_img (numpy array): version of img with JPEG artifacts
    """
    #get largest portion of image with size div by 8:
    max_x = int(img.shape[0]/8) * 8
    max_y = int(img.shape[1]/8) * 8
    im = img[0:max_x, 0:max_y, :]
    im_Y, im_Cb, im_Cr = im[:, :, 0], im[:, :, 1], im[:, :, 2]

    im_Y_tensor = torch.tensor(im_Y, dtype=torch.float).to(DEVICE)

    qt_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                     [12, 12, 14, 19, 26, 58, 60, 55],
                     [14, 13, 16, 24, 40, 57, 69, 56],
                     [14, 17, 22, 29, 51, 87, 80, 62],
                     [18, 22, 37, 56, 68, 109, 103, 77],
                     [24, 35, 55, 64, 81, 104, 113, 92],
                     [49, 64, 78, 87, 103, 121, 120, 101],
                     [72, 92, 95, 98, 112, 100, 103, 99]])
    qt_Y = np.tile(qt_Y, (int(im_Y.shape[0]/8), int(im_Y.shape[1]/8)))
    qt_Y = torch.tensor(qt_Y.astype(np.float32)*3).to(DEVICE)

    # get dct coefficients of uncompressed image
    Y_dct = torch.empty_like(im_Y_tensor).to(DEVICE)
    for i in range(0, im_Y.shape[0], 8):
        for j in range(0, im_Y.shape[0], 8):
            Y_dct[i:i+8, j:j+8] = dct_2d(im_Y_tensor[i:i+8, j:j+8], norm='ortho')

    # get JPEG compressed version of image - this will be input to the ARCNN
    Y_dct_quantized = torch.round((((Y_dct/qt_Y))))*qt_Y
    Y_dct_inv = torch.empty_like(Y_dct, dtype=torch.float).to(DEVICE)
    for i in range(0, im_Y.shape[0], 8):
        for j in range(0, im_Y.shape[0], 8):
            Y_dct_inv[i:i+8, j:j+8] = idct_2d(Y_dct_quantized[i:i+8, j:j+8], norm='ortho')

    compressed_img = np.zeros(im.shape)
    compressed_img[:, :, 0] = Y_dct_inv.detach().cpu().numpy()
    compressed_img[:, :, 1] = im_Cb
    compressed_img[:, :, 2] = im_Cr
    compressed_img = ycbcr2rgb(compressed_img)
    return compressed_img
