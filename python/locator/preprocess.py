import os
import cv2
import shutil
import numpy as np
from PIL import ImageDraw, Image
from utils.util import mkdir


def binary_fillhole(img):
    # blackhat
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (33, 33), anchor=(16, 16))
    img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, element)
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # otsu binary
    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=img)
    # fill small holes
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)
    return img


def find_region(row_value_threshold, col_value_threshold, row_range_threshold, col_range_threshold, need_range):
    global h, w
    row_candidate = np.argwhere(row_sum > row_value_threshold * w * 255)
    col_candidate = np.argwhere(col_sum > col_value_threshold * h * 255)
    row_start = col_start = 0
    row_end = h
    col_end = w
    if need_range:
        if row_candidate.size > 0:
            start_candidate = row_candidate[row_candidate < row_range_threshold[0] * h]
            end_candidate = row_candidate[row_candidate > row_range_threshold[1] * h]
            if start_candidate.size > 0:
                row_start = start_candidate[0]
            if end_candidate.size > 0:
                row_end = end_candidate[0]
        if col_candidate.size > 0:
            start_candidate = col_candidate[col_candidate < col_range_threshold[0] * w]
            end_candidate = col_candidate[col_candidate > col_range_threshold[1] * w]
            if start_candidate.size > 0:
                col_start = start_candidate[0]
            if end_candidate.size > 0:
                col_end = end_candidate[0]
    else:
        row_start, row_end = row_candidate[0][0], row_candidate[-1][0]
        col_start, col_end = col_candidate[0][0], col_candidate[-1][0]
    return row_start, col_start, row_end, col_end


# \\ is requried at the end of the path
img_dir = r"c:\Users\Deng\Desktop\difangzhi\out\\"
images = [img_dir + f.name for f in os.scandir(img_dir) if f.name.endswith("jpg")]
images.sort(key=lambda item: (len(item), item))
mkdir("bw_out")
mkdir("color_out")
for i, image_path in enumerate(images):
    text_path = img_dir + image_path.split('\\')[-1][: -3] + 'txt'
    color_img = cv2.imread(image_path)
    img = np.copy(color_img)
    img = binary_fillhole(img)
    h, w = img.shape
    row_sum = np.sum(img, 1)
    col_sum = np.sum(img, 0)
    row_start, col_start, row_end, col_end = find_region(0.5, 0.5, (0.35, 0.7), (0.35, 0.9), True)
    if row_start == 0 or col_start == 0:
        row_start, col_start, row_end, col_end = find_region(0.01, 0.02, (0, 0), (0, 0), False)
    new_lines = []
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split()
            x1, y1, x2, y2 = map(int, line_split[0: 4])
            x1 -= col_start
            y1 -= row_start
            x2 -= col_start
            y2 -= row_start
            new_line = str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + ' '
            for s in line_split[4: -1]:
                new_line += s + ' '
            new_line += line_split[-1] + '\n'
            new_lines.append(new_line)
    with open('./bw_out/{}.txt'.format(i), 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    img = img[row_start: row_end, col_start: col_end]
    color_img = color_img[row_start: row_end, col_start: col_end]
    cv2.imwrite("./bw_out/{}.jpg".format(i), img)
    cv2.imwrite("./color_out/{}.jpg".format(i), color_img)
    shutil.copyfile('./bw_out/{}.txt'.format(i), './color_out/{}.txt'.format(i))
