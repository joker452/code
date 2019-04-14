import cv2
import numpy as np


def my_filter(image, is_vertical, x_min, x_max, y_min, y_max, threshold, min_length):
    projection = []
    interval = []
    if is_vertical:
        for x in range(x_min, x_max):
            r = image[y_min: y_max, x, :].mean(1).mean()
            if r < threshold:
                r = 0
                image[y_min: y_max, x, :] = 0
            projection.append(r)
        begin = length = 0
        scan = False
        for i, p in enumerate(projection):
            if p != 0:
                if not scan:
                    begin = i
                    scan = True
                length += 1
            else:
                if length >= min_length:
                    interval.append((begin, length))
                else:
                    image[y_min: y_max, x_min + i - length - 1: x_min + i - 1, :] = 0
                begin = 0
                length = 0
                scan = False
    else:
        for y in range(y_min, y_max):
            r = image[y, x_min: x_max, :].mean(1).mean()
            if r < threshold:
                r = 0
                image[y, x_min: x_max, :] = 0
            projection.append(r)

        begin = length = 0
        scan = False
        for i, p in enumerate(projection):
            if p != 0:
                if not scan:
                    begin = i
                    scan = True
                length += 1
            else:
                if length >= min_length:
                    interval.append((begin, length))
                else:
                    image[y_min + i - length - 1: y_min + i - 1, x_min: x_max, :] = 0
                begin = 0
                length = 0
                scan = False
    return interval
def fluctuation(image, is_vertical):
    f = 0
    h, w, _ = image.shape
    if is_vertical:
        for x in range(w):
            i = image[0, x, :].mean()
            for y in range(h):
                if i != image[y, x, :].mean():
                    f += 1
        f = f / w
    else:
        for y in range(h):
            i = image[y, 0, :].mean()
            for x in range(w):
                if i != image[y, x, :].mean():
                    f += 1
        f = f / h
    return f
def extract(img, bounding_rects):
    images = []
    for box in bounding_rects:
        x, y, width ,height = list(map(int, box))
        images.append(np.copy(img[y: y + height, x: x+width, :]))
    return images
img = cv2.imread('1.jpg')
h, w, _ = img.shape
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# cv2.imwrite('a.png', img)
# cv2.imshow('1', img)
dst = cv2.morphologyEx(img, 6, cv2.getStructuringElement(1, (33, 33), (16, 16)))
dst2 = dst.mean(2)[:, :, None]
dst = np.where(dst2 < 50, (0, 0, 0), (255, 255, 255))
cv2.imwrite('before.jpg', dst)
# threshold = 15
# min_length = 15
# bounding_rects = []
# v_intervals = my_filter(dst, True, 0, w, 0, h, threshold, min_length)
# for v_interval in v_intervals:
#     h_intervals = my_filter(dst, False, v_interval[0], v_interval[0] + v_interval[1], 0, h, threshold, min_length)
#     for h_interval in h_intervals:
#         factor = 0.3
#         expand_x = v_interval[1] * factor
#         expand_y = h_interval[1] * factor
#         width = v_interval[1] + expand_y
#         height = h_interval[1] + expand_x
#         x = v_interval[0] - 0.5 * expand_y
#         y = h_interval[0] - 0.5 * expand_x
#         if x < 0:
#             x = 0
#         if x > w:
#             x = w - 1 -width
#         if y < 0:
#             y = 0
#         if y > h:
#             y = h - 1- height
#         bounding_rects.append((x, y, width, height))
# images = extract(dst, bounding_rects)
# j = 0
# for i, character in enumerate(images):
#     # nonzero = []
#     # h, w, _ = character.shape
#     # for x in range(w):
#     #     for y in range(h):
#     #         if character[y, x, :].mean() != 0:
#     #             nonzero.append(cv2.(bounding_rects[i][0] + x, bounding_rects[i][1] + y))
#     if fluctuation(character, True) <= 1 or fluctuation(character, False) <= 1:
#         del images[j]
#         del bounding_rects[j]
#         j -= 1
#     j += 1
# cv2.imwrite('after.jpg', dst)
