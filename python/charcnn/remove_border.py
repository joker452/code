import os
import cv2
import numpy as np
img_dir = r"c:\users\deng\desktop\out\\"
images = [img_dir + f.name for f in os.scandir(img_dir) if f.name.endswith("jpg")]
images.sort(key=lambda  item: (len(item), item))
for i, image_path in enumerate(images):
    text_path = img_dir + image_path.split('\\')[-1][: -3] + 'txt'
    img = cv2.imread(image_path)
    img = img[:, :, 0]
    h, w = img.shape
    img = np.where(img > 200, 255, 0)
    row_start = col_start = 0
    row_end = h
    col_end = w
    row_sum = np.sum(img, 1)
    col_sum = np.sum(img, 0)
    row_candidate = np.argwhere(row_sum > 0.5 * w * 255)
    col_candidate = np.argwhere(col_sum > 0.5 * h * 255)
    if len(row_candidate) > 0:
        for r in row_candidate:
            if r[0] < 0.35 * h:
                row_start = r[0]
                break
        for r in reversed(row_candidate):
            if r[0] > 0.7 * h:
                row_end = r[0]
                break
    if len(col_candidate) > 0:
        for c in col_candidate:
            if c[0] < 0.35 * w:
                col_start = c[0]
                break
        for c in col_candidate:
            if c[0] > 0.9 * w:
                col_end = c[0]
                break
    img = img[row_start: row_end, col_start: col_end]
    # h, w = img.shape
    # image = np.zeros((h, w, 3))
    # image[:, :, 0] = img
    # image[:, :, 1] = img
    # image[:, :, 2] = img
    new_lines = []
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            x1, y1, x2, y2 = map(int, line.split())
            x1 -= col_start
            y1 -= row_start
            x2 -= col_start
            y2 -= row_start
            new_line = str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n'
            new_lines.append(line)
    with open('./label/{}.txt'.format(i), 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    # cv2.imwrite("./out/{}.jpg".format(i), image)



