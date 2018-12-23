import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CASIA(Dataset):

    def __init__(self, image_dir, label_dir, label_files, char_class):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)
        self.image_list.sort()
        self.label_dir = label_dir
        self.label_list = os.listdir(label_dir)
        self.label_list.sort()
        char_list = []
        for label_file in label_files:
            with open(label_file, 'r', encoding='cp936') as f:
                file_lines = f.read().split('\n')[1: -1]
                for line in file_lines:
                    char_code = line.split()[1]
                    if len(char_code) < 6:
                        char_code = char_code[0: 2] + '00' + char_code[2:]
                    char_list.append(char_code)
        if len(char_list) != char_class:
            raise ValueError(
                "Classes from files don't equal char_class, expected {}, got {}".format(char_class, len(char_list)))
        self.char2index = dict(zip(char_list, range(len(char_list))))
        print(len(char_list))
        # change this line for different size of input
        self.transforms = transforms.Compose([transforms.ToTensor()])
        print(len(self.image_list))

    def get_image(self, img_path, img_name):
        # crop the character from the original image
        img = cv2.imread(os.path.join(img_path, img_name), 0)
        black_index = np.where(img < 255)
        try:
            min_x = min(black_index[0])
            max_x = max(black_index[0])
            min_y = min(black_index[1])
            max_y = max(black_index[1])
        except ValueError:
            print(os.path.join(img_path, img_name))
            img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img[min_x:max_x, min_y:max_y], dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        # put the resized character into the center of the image background with target size
        # h, w = img.shape
        # img_bg = np.full((114, 114), 255, dtype='uint8')
        # row_start = (img_bg.shape[0] - h) // 2
        # col_start = (img_bg.shape[1] - w) // 2
        # img_bg[row_start: row_start + h, col_start: col_start + w] = img
        return Image.fromarray(img)

    def __getitem__(self, index):
        image = self.get_image(self.image_dir, self.image_list[index])
        image = self.transforms(image)

        with open(os.path.join(self.label_dir, self.label_list[index]), 'r', encoding='utf-8') as f:
            char_code = f.read()
        label = torch.zeros(7356, dtype=torch.float)
        label[self.char2index[char_code]] = 1.0
        # label = torch.tensor(self.char2index[char_code], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.image_list)
