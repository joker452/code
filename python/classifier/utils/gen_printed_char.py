from __future__ import print_function
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import pickle
import argparse
import struct
import os
import cv2
import random
import numpy as np


class dataAugmentation(object):
    def __init__(self, noise=True, dilate=True, erode=True):
        self.noise = noise
        self.dilate = dilate
        self.erode = erode

    def add_noise(selfs, img, threshold=0.4):
        for i in range(int(img.shape[0] * image.shape[1] * 0.4)):  # 添加点噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 255
        return img

    def add_erode(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.erode(img, kernel)
        return img

    def add_dilate(self, img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel)
        return img

    def do(self, img):
        if self.noise and random.random() < 0.5:
            img = self.add_noise(img)
        if self.dilate and random.random() < 0.5:
            img = self.add_dilate(img)
        elif self.erode:
            img = self.add_erode(img)
        return img


# 对字体图像做等比例缩放
class PreprocessResizeKeepRatio(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def do(self, np_img):
        max_width = self.width
        max_height = self.height

        cur_height, cur_width = np_img.shape[:2]

        ratio_w = float(max_width) / float(cur_width)
        ratio_h = float(max_height) / float(cur_height)
        ratio = min(ratio_w, ratio_h)

        new_size = (int(cur_width * ratio), int(cur_height * ratio))

        new_size = (max(new_size[0], 1), max(new_size[1], 1),)

        resized_img = cv2.resize(np_img, new_size)
        return resized_img


# 查找字体的最小包含矩形
class FindImageBBox(object):
    def __init__(self):
        pass

    def do(self, img):
        row, coloum = np.where(img)
        l = min(coloum)
        r = max(coloum)
        u = min(row)
        d = max(row)
        return l, u, r, d


# 把字体图像放到背景图像中
class PreprocessResizeKeepRatioFillBG(object):

    def __init__(self, width, height, fill_bg=False, auto_avoid_fill_bg=True, margin=None):
        self.width = width
        self.height = height
        self.fill_bg = fill_bg
        self.auto_avoid_fill_bg = auto_avoid_fill_bg
        self.margin = margin

    def is_need_fill_bg(self, np_img):
        height, width = np_img.shape
        if height * 3 < width:
            return True
        if width * 3 < height:
            return True
        return False

    def put_img_into_center(self, img_large, img_small, ):
        width_large = img_large.shape[1]
        height_large = img_large.shape[0]

        width_small = img_small.shape[1]
        height_small = img_small.shape[0]

        if width_large < width_small:
            raise ValueError("width_large <= width_small")
        if height_large < height_small:
            raise ValueError("height_large <= height_small")

        start_width = (width_large - width_small) // 2
        start_height = (height_large - height_small) // 2

        img_large[start_height:start_height + height_small, start_width:start_width + width_small] = img_small
        return img_large

    def do(self, np_img):
        # 确定有效字体区域，原图减去边缘长度就是字体的区域
        if self.margin is not None:
            width_minus_margin = max(2, self.width - 2 * self.margin)
            height_minus_margin = max(2, self.height - 2 * self.margin)
        else:
            width_minus_margin = self.width
            height_minus_margin = self.height

        if len(np_img.shape) > 2:
            pix_dim = np_img.shape[2]
        else:
            pix_dim = None

        resize = PreprocessResizeKeepRatio(width_minus_margin, height_minus_margin)
        resized_np_img = resize.do(np_img)

        if self.auto_avoid_fill_bg:
            # fill_bg在长宽或者宽长比大于3时为True
            self.fill_bg = self.is_need_fill_bg(np_img)

        # should skip horizontal stroke
        if not self.fill_bg:
            # 不填充背景就直接resize到目标大小
            ret_img = cv2.resize(resized_np_img, (width_minus_margin, height_minus_margin))
        else:
            # 否则生成目标大小的白色背景
            if pix_dim is not None:
                norm_img = np.zeros((height_minus_margin, width_minus_margin, pix_dim), np.uint8)
            else:
                norm_img = np.zeros((height_minus_margin, width_minus_margin), np.uint8)
            # 将缩放后的字体图像置于背景图像中央
            ret_img = self.put_img_into_center(norm_img, resized_np_img)

        if self.margin is not None:
            # 放到最终有边缘的大图片中去
            if pix_dim is not None:
                norm_img = np.zeros((self.height, self.width, pix_dim), np.uint8)
            else:
                norm_img = np.zeros((self.height, self.width), np.uint8)
            ret_img = self.put_img_into_center(norm_img, ret_img)
        return ret_img


# 检查字体文件是否可用
class FontCheck(object):

    def __init__(self, char_dict, width=32, height=32):
        self.char_dict = char_dict
        self.width = width
        self.height = height

    def do(self, font_path):
        width = self.width
        height = self.height
        for (gbkcode, value) in self.char_dict.items():  # 对汉字循环
            gbkcode = gbkcode[:2] + gbkcode[-2:] + gbkcode[2: 4]
            char = struct.pack('>H', int(gbkcode, base=16)).decode('gbk')
            img = Image.new("L", (width, height), "black")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, int(min(self.width, self.height) * 0.7))
            draw.text((0, 0), char, "white", font=font)
            data = np.asarray(img, dtype='uint8')
            if data.sum() < 255 * 2:
                # There aren't enough white pixels to show the font can be used for all characters
                return False
        return True


# 生成字体图像
class Font2Image(object):

    def __init__(self, width, height, margin):
        self.width = width
        self.height = height
        self.margin = margin

    def do(self, font_path, char, rotate=0):
        find_image_bbox = FindImageBBox()
        # 黑色背景
        img = Image.new("L", (self.width, self.height), "black")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(min(self.width, self.height) * 0.7))
        # 白色字体
        draw.text((0, 0), char, 'white', font=font)
        if rotate != 0:
            img = img.rotate(rotate, expand=False)
        np_img = np.asarray(img, dtype='uint8')
        left, upper, right, lower = find_image_bbox.do(np_img)
        np_img = np_img[upper: lower + 1, left: right + 1]
        preprocess = PreprocessResizeKeepRatioFillBG(self.width, self.height, fill_bg=False, margin=self.margin)
        np_img = preprocess.do(np_img)
        # 重新二值化并反转字体颜色和背景
        np_img = np.where(np_img > 150, 255, 0).astype(dtype='uint8')
        np_img = np.full_like(np_img, 255) - np_img
        return np_img


# 注意，chinese_labels里面的映射关系是：（gbkcode：index）
def get_label_dict(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def args_parse():
    # 解析输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', dest='out_dir', required=True, help='dir for output images')
    parser.add_argument('--font_dir', dest='font_dir', help='font dir to to check all the fonts in it')
    parser.add_argument('--width', default=None, type=int, required=True, help='width')
    parser.add_argument('--height', default=None, type=int, required=True, help='height')
    parser.add_argument('--margin', dest='margin', type=int, default=0)
    parser.add_argument('--rotate', dest='rotate', default=0, type=int, help='max rotate degree 0-45')
    parser.add_argument('--char', '-c', dest='char', required=True, help='The target chinese character')
    parser.add_argument('--dict_path', '-dict', dest='dict_path', help='Path to the dictionary pkl')
    args = vars(parser.parse_args())
    return args


def checkfonts(font_dir, dict_path, width, height):
    # 对于每类字体进行小批量测试
    verified_font = {}
    label_dict = get_label_dict(dict_path)
    check = FontCheck(label_dict, width, height)
    for font_name in os.listdir(font_dir):
        if check.do(os.path.join(font_dir, font_name)):
            verified_font[font_name] = True
        else:
            verified_font[font_name] = False
    print(verified_font)


def generate_character(character, width, height, margin, font_path, rotate):
    font2image = Font2Image(width, height, margin)
    image = font2image.do(font_path, character, rotate)
    return Image.fromarray(image)


if __name__ == "__main__":
    options = args_parse()
    # note that margin is important to place the character in the center of the final image
    out_dir = os.path.expanduser(options['out_dir'])
    font_dir = options['font_dir']
    width = options['width']
    height = options['height']
    margin = options['margin']
    rotate = options['rotate']
    need_aug = options['need_aug']
    dict_path = options['dict_path']
    char = options['char']

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    font_list = os.listdir(font_dir)
    font2image = Font2Image(width, height, margin)
    if rotate < 0:
        roate = - rotate
    rotate = rotate % 360
    arg = dataAugmentation()
    for index, font in enumerate(font_list):
        image = font2image.do(os.path.join(font_dir, font), char, rotate)
        cv2.imwrite(os.path.join(out_dir, '{}.png'.format(index)), image)
