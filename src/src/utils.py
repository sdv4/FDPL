"""
   Functions of general utility that are used throughout the project.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from PIL import Image

def ycbcr2rgb(im):
    """
        Takes images in YCbCr format and converts it to RGB

        Args:
            im (np.ndarray): image in YCbCr format
        Returns:
            rgb (np.ndarray): im in rgb format
        source: https://stackoverflow.com/questions/34913005/color-space-mapping-ycbcr-to-rgb
    """
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def psnr(img1, img2):
    """
        Calculate the Peak Signal to Noise Ratio between two images.
        formula: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Args:
            img1 (np.ndarray): image of any number of channels
            img2 (np.ndarray): image of any number of channels

        Returns:
            psnr (float): psnr between img1 and img2 in dB
    """
    mse = np.mean(np.power(img1.astype(np.double) - img2.astype(np.double), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    p_snr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return p_snr

# note: all dct related functions are either exactly as or based on those
# at https://github.com/zh217/torch-dct
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_r = V_t_r.to(device)
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_t_i = V_t_i.to(device)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.irfft(V, 1, onesided=False)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
    """
        source: https://gist.github.com/dem123456789/23f18fd78ac8da9615c347905e64fc78
    """
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2) < patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0, 0, num_padded_H_Top, num_padded_H_Bottom), 0)
        img = padding_H(img)
    if(img.size(3) < patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right, 0, 0), 0)
        img = padding_W(img)
    step_int = [0, 0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,
                                    img[:, :, -patch_H:, :].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW, 
                                     patches_fold_H[:, :, :, -patch_W:, :].permute(0, 1, 2, 4, 3).unsqueeze(3)), dim=3)
    patches = patches_fold_HW.permute(2, 3, 0, 1, 4, 5)
    patches = patches.reshape(-1, img.size(0), img.size(1), patch_H, patch_W)
    if(batch_first):
        patches = patches.permute(1, 0, 2, 3, 4)
    return patches

# source: https://stackoverflow.com/questions/47544242/how-to-fix-a-rotated-output-in-zoom-in-region-in-image
def zoom_in_rec(input, ax, rect, cmap):
    axins = zoomed_inset_axes(ax, 2, loc=3)  
    x1, x2, y1, y2 = rect.get_x(), rect.get_x()+rect.get_width(),rect.get_y(), rect.get_y()+rect.get_height() # specify the limits
    axins.set_xlim(x1, x2)  # apply the x-limits
    axins.set_ylim(y1, y2)  # apply the y-limits
    #mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='red',lw=1.0, ls=':')
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    rot = np.flipud(input)
    axins.imshow(Image.fromarray(rot, 'YCbCr'))

def zoom_in_rec2(input, ax, rect, cmap):
    axins = zoomed_inset_axes(ax, 2, loc=3)  
    x1, x2, y1, y2 = rect.get_x(), rect.get_x()+rect.get_width(),rect.get_y(), rect.get_y()+rect.get_height() # specify the limits
    axins.set_xlim(x1, x2)  # apply the x-limits
    axins.set_ylim(y1, y2)  # apply the y-limits
    #mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='red',lw=1.0, ls=':')
    #zoomed_inset_axes(ax, zoom=0.5)
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    rot = np.flipud(input)
    axins.imshow(rot)
