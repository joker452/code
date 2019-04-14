import cv2
# import numpy as np
# src = cv2.imread('c:/Users/Deng/Desktop/out/50.jpg')
# d = np.where(src < 200, 0, 255)
# d = d.astype(np.uint8, copy=False)
# dst = cv2.morphologyEx(d, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
# # cv2.imwrite('c:/Users/Deng/Desktop/555.jpg',dst)
# with open('c:/users/deng/desktop/aa/g/1.txt', 'r', encoding='utf-8') as f:
#     a = f.readlines()
# b = []
# for i, x in enumerate(a):
#     x1, y1, x2, y2 = x.split()[1:]
#     b.append(x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + '\n')
# with open('c:/Users/Deng/Desktop/3.txt', 'w', encoding='utf-8') as f:
#     f.writelines(b)
print(cv2.imread('c:/code/python/charcnn/utils/1.jpg').shape)