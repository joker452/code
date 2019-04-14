from skimage.io import imread, imsave
import cv2
import os
import numpy as np
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
root_dir = 'd:/lunwen/data/difangzhi/data/'
img_names = os.listdir(root_dir)
img_names = [os.path.join(root_dir, f) for f in img_names]
for i, img_name in enumerate(img_names):
    img = imread(img_name)
    if img.ndim == 3:
        img = img_as_ubyte(rgb2gray(img))
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 1 for char
    th = np.where(th > 200, 0, 1)
    thr=30
    thr2=2
    h, w = th.shape
    row_hist = th.sum(1)
    border = h // 4
    pos_upper = row_hist[0: border].argmax()
    max_upper = row_hist[pos_upper]
    pos_lower = row_hist[h- border + 1: ].argmax()
    pos_lower = pos_lower + h - border + 1
    max_lower = row_hist[pos_lower]
    ave = row_hist.mean()

    if row_hist[:pos_upper].sum() < max_upper * thr and  max_upper > ave * thr2:
        while row_hist[pos_upper]>ave:
            pos_upper=pos_upper+1
    else:
        pos_upper=0
    if row_hist[pos_lower: ].sum() < max_lower * thr and max_lower > ave * thr2:
        while row_hist[pos_lower] >ave:
            pos_lower=pos_lower-1
    else:
        pos_lower = h

    col_hist = th.sum(0)

    border = w // 8
    pos_left = col_hist[:border].argmax()
    max_left = col_hist[pos_left]
    pos_right = col_hist[w - border: ].argmax()
    pos_right = pos_right + w - border
    max_right = col_hist[pos_right]

    ave = col_hist.mean()

    if  col_hist[: pos_left].sum() < max_left * thr and max_left > ave * thr2:
        while col_hist[pos_left] > ave:
            pos_left = pos_left + 1
    else:
        pos_left = 1

    if col_hist[pos_right:].sum() < max_right * thr and max_right > ave * thr2:
        while col_hist[pos_right] > ave:
            pos_right = pos_right - 1
    else:
        pos_right = w
    im = th[pos_upper:pos_lower, pos_left: pos_right]
    im = np.where(im == 1, 255, 0)
    imsave('d:/lunwen/data/a/{}.png'.format(i), im)
